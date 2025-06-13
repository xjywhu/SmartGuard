import pickle
import numpy as np
import os
import sys

# --- Configuration ---
# Base path to the data directory structure
base_data_path = 'd:\\SmartGuard\\data'
output_log_dir = 'd:\\SmartGuard\\decoded_logs'

# Define dataset files and their types ('an', 'fr', 'sp')
# This helps in selecting the correct dictionary
dataset_files_info = {
    'an': [
        os.path.join(base_data_path, 'an_data', 'an_trn_instance_10.pkl'),
        os.path.join(base_data_path, 'an_data', 'an_vld_instance_10.pkl'),
        os.path.join(base_data_path, 'an_data', 'an_test_instance_10.pkl'),
        # Add attack files if they follow the same 4-feature structure
        # os.path.join(base_data_path, 'an_data', 'attack', 'labeled_an_light_attack.pkl'), # Example
    ],
    'fr': [
        os.path.join(base_data_path, 'fr_data', 'fr_trn_instance_10.pkl'),
        os.path.join(base_data_path, 'fr_data', 'fr_vld_instance_10.pkl'),
        os.path.join(base_data_path, 'fr_data', 'fr_test_instance_10.pkl'),
        os.path.join(base_data_path, 'fr_data', 'fr_add_trn.pkl'),
        # os.path.join(base_data_path, 'fr_data', 'labeled_fr_test.pkl'), # Might have different structure
    ],
    'sp': [
        os.path.join(base_data_path, 'sp_data', 'sp_trn_instance_10.pkl'),
        os.path.join(base_data_path, 'sp_data', 'sp_vld_instance_10.pkl'),
        os.path.join(base_data_path, 'sp_data', 'sp_test_instance_10.pkl'),
        os.path.join(base_data_path, 'sp_data', 'sp_add_trn.pkl'),
    ]
}

# --- Helper function to load dictionaries ---
def load_dictionaries(dataset_type):
    """Loads dictionaries for a given dataset type ('an', 'fr', 'sp')."""
    dict_path = os.path.join(base_data_path, 'data', dataset_type, 'dictionary.py')
    if not os.path.exists(dict_path):
        print(f"Warning: Dictionary file not found for {dataset_type} at {dict_path}")
        return None, None, None

    # This is a simplified way to load dicts from a .py file.
    # A more robust way would be to ensure these .py files can be imported as modules.
    loaded_dicts = {}
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            exec(f.read(), loaded_dicts)
        # Reverse the dictionaries for easy lookup
        dcd = {v: k for k, v in loaded_dicts.get('device_control_dict', {}).items()}
        dwd = {v: k for k, v in loaded_dicts.get('dayofweek_dict', {}).items()}
        hd = {v: k for k, v in loaded_dicts.get('hour_dict', {}).items()}
        return dcd, dwd, hd
    except Exception as e:
        print(f"Error loading dictionaries from {dict_path}: {e}")
        return None, None, None

# --- Main processing ---
if not os.path.exists(output_log_dir):
    os.makedirs(output_log_dir)
    print(f"Created output directory: {output_log_dir}")

for dataset_type, file_list in dataset_files_info.items():
    print(f"\nProcessing dataset type: {dataset_type.upper()}")
    reverse_device_control_dict, reverse_dayofweek_dict, reverse_hour_dict = load_dictionaries(dataset_type)

    if not reverse_device_control_dict or not reverse_dayofweek_dict or not reverse_hour_dict:
        print(f"Skipping {dataset_type} due to missing dictionaries.")
        continue

    for data_file_path in file_list:
        if not os.path.exists(data_file_path):
            print(f"  File not found, skipping: {data_file_path}")
            continue

        print(f"  Processing file: {data_file_path}")
        output_file_name = os.path.basename(data_file_path).replace('.pkl', '_decoded.txt')
        output_file_path = os.path.join(output_log_dir, output_file_name)

        try:
            with open(data_file_path, 'rb') as f_in, open(output_file_path, 'w', encoding='utf-8') as f_out:
                all_sequences = pickle.load(f_in)
                f_out.write(f"Decoded logs for: {data_file_path}\n")
                f_out.write(f"Total sequences: {len(all_sequences)}\n\n")

                for seq_idx, sequence_raw_list in enumerate(all_sequences):
                    sequence_raw = np.array(sequence_raw_list)
                    f_out.write(f"Sequence {seq_idx + 1}:\n")

                    sequence_length = 10
                    num_features = 4

                    if sequence_raw.size == sequence_length * num_features:
                        reshaped_sequence = sequence_raw.reshape(sequence_length, num_features)
                        for event_idx, event_features in enumerate(reshaped_sequence):
                            day_id = int(event_features[0])
                            hour_id = int(event_features[1])
                            duration_id = int(event_features[2]) # Meaning of duration ID might still be abstract
                            device_control_id = int(event_features[3])

                            day_str = reverse_dayofweek_dict.get(day_id, f"UnknownDayID:{day_id}")
                            hour_str = reverse_hour_dict.get(hour_id, f"UnknownHourID:{hour_id}")
                            action_str = reverse_device_control_dict.get(device_control_id, f"UnknownActionID:{device_control_id}")

                            f_out.write(f"  Event {event_idx + 1}: Day='{day_str}', Hour='{hour_str}', DurationID={duration_id}, Action='{action_str}'\n")
                    elif sequence_raw.size > 0 and sequence_raw.ndim == 1 and num_features == 1: # Handling for files like fr_add_trn.pkl if they are just lists of device_control_ids
                        f_out.write(f"  (Note: Sequence appears to be 1D, assuming device_control_ids only)\n")
                        for event_idx, device_control_id_only in enumerate(sequence_raw):
                            action_str = reverse_device_control_dict.get(int(device_control_id_only), f"UnknownActionID:{int(device_control_id_only)}")
                            f_out.write(f"  Event {event_idx + 1}: Action='{action_str}'\n")
                    else:
                        f_out.write(f"  Error: Sequence {seq_idx + 1} has an unexpected size of {sequence_raw.size}. Expected {sequence_length * num_features} or 1D array. Raw: {sequence_raw.tolist()}\n")
                    f_out.write("-\n") # Separator for sequences

        except FileNotFoundError:
            print(f"    Error: Data file not found during processing: {data_file_path}")
        except Exception as e:
            print(f"    Error processing {data_file_path}: {e}")
        else:
            print(f"    Successfully decoded and saved to: {output_file_path}")

print("\nAll processing finished.")