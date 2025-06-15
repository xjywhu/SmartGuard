import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random
from collections import Counter


def find_next_occurrence_at_indices(arr, target, indices, i):
    first_occurrence = None

    for t in indices:
        if 0 <= t < len(arr) and arr[t] == target:
            first_occurrence = t + i + 1
            break
        first_occurrence = indices[-1]

    return first_occurrence

def _compute_duration_embeddings(input_ids: torch.Tensor, embedding_dim) -> torch.Tensor:
    persistence_indices = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
    seq_len = 40
    duration_embeddings = torch.zeros(int(seq_len/4), device=input_ids.device)
    time_embeddings = torch.zeros(int(seq_len/4), device=input_ids.device)
    day_embeddings = input_ids[0:37:4].tolist()
    hour_embeddings = input_ids[1:38:4].tolist()
    time_embeddings[0] = 0
    for i in range(len(day_embeddings)-1):
        if day_embeddings[i] * 24 + hour_embeddings[i] * 3 <= day_embeddings[i+1] * 24 + \
                hour_embeddings[i+1] * 3:
            time_embeddings[i + 1] = day_embeddings[i + 1] * 24 + hour_embeddings[i + 1] * 3 - day_embeddings[i] * 24 - hour_embeddings[i] * 3 + time_embeddings[i]
        else:
            time_embeddings[i + 1] = day_embeddings[i + 1] * 24 + hour_embeddings[i + 1] * 3 - day_embeddings[i] * 24 - \
                                     hour_embeddings[i] * 3 + 168 + time_embeddings[i]

    input_seq = input_ids
    behavior_list = []
    intro_time = []
    for idx in persistence_indices:
        behavior_list.append(input_seq[idx])
    indices = [t for t in range(0, len(behavior_list))]
    for i in range(len(behavior_list)):
        next_arr = behavior_list[i + 1:]
        next_index = find_next_occurrence_at_indices(next_arr, behavior_list[i], indices, i)
        intro_time.append(next_index)

    for i in range(len(intro_time)):
            duration_embeddings[i] = time_embeddings[intro_time[i]] - time_embeddings[i]


    scaled_duration = (duration_embeddings.unsqueeze(1) * torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float, device=input_ids.device) * -(
                    np.log(10000.0) / embedding_dim))).float()
    # scaled_duration = torch.tensor(scaled_duration)
    scaled_duration = scaled_duration.clone().detach()
    sinusoidal_duration_embedding = torch.zeros(int(seq_len/4), embedding_dim, device=input_ids.device)
    sinusoidal_duration_embedding[:, 0::2] = scaled_duration
    sinusoidal_duration_embedding[:, 1::2] = scaled_duration

    return sinusoidal_duration_embedding

class TimeSeriesDataset(Dataset):
    def __init__(self, data, embedding_dim):
        self.data = data
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # device_control = sample.reshape(10, 4).T[3]
        device_control = sample.reshape(10, 4).T[3]
        # device_control = device_control.reshape(10, 1)

        encoder_input = sample
        decoder_output = device_control

        encoder_input = torch.from_numpy(encoder_input)
        decoder_output = torch.from_numpy(decoder_output)
        duration_input = _compute_duration_embeddings(encoder_input, self.embedding_dim)

        return encoder_input, decoder_output, duration_input


class PegasusSinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int) -> None:

        super().__init__(num_positions, embedding_dim)

        self.persistence_indices = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]  # 关注的索引位置
        self.hour_indices = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37]
        self.day_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
        self.day_weight = nn.Parameter(torch.tensor([0.4]), requires_grad=True)
        self.hour_weight = nn.Parameter(torch.tensor([0.4]), requires_grad=True)
        self.duration_weight = nn.Parameter(torch.tensor([0.7]), requires_grad=True)
        self.order_weight = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        # self.time_indices = [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37]

    def find_next_occurrence_at_indices(self, arr, target, indices, i):
        first_occurrence = None

        for t in indices:
            if 0 <= t < len(arr) and arr[t] == target:
                first_occurrence = t + i + 1
                break
            first_occurrence = indices[-1]

        return first_occurrence

    def _compute_order_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        # order_embeddings = torch.zeros(bsz, int(seq_len/4), device=input_ids.device)

        order_embeddings = torch.arange(0, int(seq_len/4)).unsqueeze(1).repeat(bsz, 1).reshape(bsz, int(seq_len/4))
        order_embeddings = order_embeddings.to(input_ids.device)

        scaled_order = (order_embeddings.unsqueeze(2) * torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=input_ids.device) * -(
                    np.log(10000.0) / self.embedding_dim))).float()
        sinusoidal_order_embedding = torch.zeros(bsz, int(seq_len/4), self.embedding_dim, device=input_ids.device)
        sinusoidal_order_embedding[:, :, 0::2] = torch.sin(scaled_order)
        sinusoidal_order_embedding[:, :, 1::2] = torch.cos(scaled_order)

        return sinusoidal_order_embedding

    def _compute_day_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        day_embeddings = input_ids[:, 0:37:4]

        scaled_day = (day_embeddings.unsqueeze(2) * torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=input_ids.device) * -(
                        np.log(10000.0) / self.embedding_dim))).float()
        sinusoidal_day_embedding = torch.zeros(bsz, int(seq_len/4), self.embedding_dim, device=input_ids.device)
        sinusoidal_day_embedding[:, :, 0::2] = torch.sin(scaled_day)
        sinusoidal_day_embedding[:, :, 1::2] = torch.cos(scaled_day)

        return day_embeddings, sinusoidal_day_embedding

    def _compute_hour_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        hour_embeddings = input_ids[:, 1:38:4]

        # 正弦余弦化持续性嵌入
        scaled_hour = (hour_embeddings.unsqueeze(2) * torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=input_ids.device) * -(
                        np.log(10000.0) / self.embedding_dim))).float()
        sinusoidal_hour_embedding = torch.zeros(bsz, int(seq_len/4), self.embedding_dim, device=input_ids.device)
        sinusoidal_hour_embedding[:, :, 0::2] = torch.sin(scaled_hour)
        sinusoidal_hour_embedding[:, :, 1::2] = torch.cos(scaled_hour)

        return hour_embeddings, sinusoidal_hour_embedding

    # @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, sinusoidal_duration_embedding) -> torch.Tensor:
        _, sinusoidal_day_embedding = self._compute_day_embeddings(input_ids)
        _, sinusoidal_hour_embedding = self._compute_hour_embeddings(input_ids)
        sinusoidal_duration_embedding = sinusoidal_duration_embedding.to(input_ids.device)
        sinusoidal_order_embedding = self._compute_order_embeddings(input_ids)

        sinusoidal_day_embedding *= self.day_weight
        sinusoidal_hour_embedding *= self.hour_weight
        sinusoidal_duration_embedding *= self.duration_weight
        sinusoidal_order_embedding *= self.order_weight
        # return sinusoidal_order_embedding +sinusoidal_day_embedding + sinusoidal_hour_embedding + sinusoidal_duration_embedding
        return sinusoidal_order_embedding +sinusoidal_day_embedding + sinusoidal_hour_embedding + sinusoidal_duration_embedding


class SmartGuard(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, mask_strategy, mask_ratio, mask_step):
        super(SmartGuard, self).__init__()
        self.nhead = nhead
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PegasusSinusoidalPositionalEmbedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.mask_strategy = mask_strategy
        self.mask_ratio = mask_ratio
        self.mask_step = mask_step
        self.fc = nn.Linear(d_model, vocab_size)
        self.TTPE_flag = True  # Enable Three-level Time-aware Position Embedding by default

    def forward(self, x, loss_vector, epoch, duration_emb):
        # Ensure x is of shape (sequence_length, batch_size)

        seq_len = 10
        k = int(seq_len*self.mask_ratio)

        tgt_mask = []
        device_control = x.reshape(x.size(0), 10, 4).T[3].T
        # if not loss_vector:

        if self.mask_strategy == "random" or (self.mask_strategy == "top_k_loss" and epoch == 0):
            for item in device_control:
                mask = torch.ones(seq_len, seq_len)
                numbers = list(range(10))
                unique_numbers = random.sample(numbers, k)
                for idx in unique_numbers:
                    mask.T[idx] = 0

                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                tgt_mask.append(mask)
            tmp_mask = torch.stack(tgt_mask).to(x.device)
            tgt_mask = torch.repeat_interleave(tmp_mask, self.nhead, dim=0)
        else:
            if epoch <= self.mask_step:
                for item in device_control:
                    mask = self.generate_square_subsequent_mask(seq_len)
                    tgt_mask.append(mask)
                tmp_mask = torch.stack(tgt_mask).to(x.device)
                tgt_mask = torch.repeat_interleave(tmp_mask, self.nhead, dim=0)
            else:
                for item in device_control:
                    losses = []
                    for (idx, be) in enumerate(item):
                        be = be.item()
                        losses.append((loss_vector[be], idx, be))
                    losses = sorted(losses, reverse=True)

                    mask = torch.ones(seq_len, seq_len)
                    count = 0
                    for tmp in losses:
                        if count == k:
                            break
                        idx = tmp[1]
                        # mask[idx] = 0
                        mask.T[idx] = 0
                        count += 1
                    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                    # mask = mask.float().masked_fill(mask == 0, True).masked_fill(mask == 1, False)
                    tgt_mask.append(mask)

                    # print(losses)

                tmp_mask = torch.stack(tgt_mask).to(x.device)
                # print(tgt_mask.shape)

                tgt_mask = torch.repeat_interleave(tmp_mask, self.nhead, dim=0)
            # print(tgt_mask.shape)
            # break


        # input_emb = self.embedding(x)
        input_emb = self.embedding(device_control)

        if self.TTPE_flag:
            pos_emb = self.pos_embedding(x, duration_emb)
            # # pos_emb = pos_emb[:, 3:40:4, :]
            x = input_emb + pos_emb
        else:
            x = input_emb

        output = self.transformer(x, x, src_mask=tgt_mask, tgt_mask=tgt_mask)
        output = self.fc(output)

        return output, tmp_mask

    def evaluate(self, x, duration_emb):
        device_control = x.reshape(x.size(0), 10, 4).T[3].T
        input_emb = self.embedding(device_control)

        if self.TTPE_flag:
            pos_emb = self.pos_embedding(x, duration_emb)
            # pos_emb = pos_emb[:, 3:40:4, :]
            x = input_emb + pos_emb
        else:
            x = input_emb


        output = self.transformer(x, x)
        output = self.fc(output)

        return output

    def generate_square_subsequent_mask(self, sz):
        # Generate a square mask with shape (sz, sz)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

        # Convert mask values to float and apply masking
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float(0.0))

        return mask
