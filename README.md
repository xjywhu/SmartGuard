# Make Your Home Safe: Time-aware Unsupervised User Behavior Anomaly Detection in Smart Homes via Loss-guided Mask


[Jingyu Xiao](https://whalexiao.github.io/), [Zhiyao Xu](http://yuh-yang.github.io), [Qingsong Zou](#), [Qing Li](#) (*Correspondence )


-----


## Introduction
Smart homes, powered by the Internet of Things, offer great convenience but also pose security concerns due to abnormal behaviors, such as improper operations of users and potential attacks from malicious attackers. Several behavior modeling methods have been proposed to identify abnormal behaviors and mitigate potential risks. However, their performance often falls short because they do not consider temporal context, effectively learn less frequent behaviors, or account for the impact of noise in human behaviors. In this paper, we propose SmartGuard, an autoencoder-based unsupervised user behavior anomaly detection framework. First, we propose Three-level Time-aware Position Embedding (TTPE) to incorporate temporal information into positional embedding to detect temporal context anomaly. Second, we design a Loss-guided Dynamic Mask Strategy (LDMS) to encourage the model to learn less frequent behaviors, which are often overlooked during learning. Third, we propose a Noise-aware Weighted Reconstruction Loss (NWRL) to assign different weights for routine behaviors and noise behaviors to mitigate the impact of noise behaviors. Comprehensive experiments on four datasets with ten types of anomaly behaviors demonstrates that SmartGuard consistently outperforms state-of-the-art baselines and also offers highly interpretable results.

#### 0. Environment Update: 

The lightweight training requires PyTorch 2.1+, so we need to update corresponding libraries: 

```shell
# if you have set up the env for GraphGPT earlier
pip uninstall torch
pip uninstall torchvision
pip uninstall torchaudio
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# update pyg for the PyTorch 2.1+
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# install lightning
pip install lightning
```

#### 1. Update the Graph Data

Due to compatibility issues, if you are using the previously released graph data, we recommend downloading and updating it according to the provided link: [updated graph data](https://huggingface.co/datasets/Jiabin99/All_pyg_graph_data).

#### 2. Run the Baselines

You can run the scripts as follow:

**Stage-1:**

```shell
python other_baseline.py
```

**Stage-2:**

```
cd path/to/GraphGPT
sh ./scripts/tune_script/graphgpt_stage2.sh
```


## Brief Introduction 


For more technical details, kindly refer to the [paper](https://arxiv.org/abs/2310.13023) and the project [website](https://graphgpt.github.io/) of our Graph. 


-----------

<span id='Usage'/>

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment Preparation'>2. Environment Preparation </a>
* <a href='#Training GraphGPT'>3. Training GraphGPT </a>
  * <a href='#Prepare Pre-trained Checkpoint'>3.1. Prepare Pre-trained Checkpoint</a>
  * <a href='#Self-Supervised Instruction Tuning'>3.2. Self-Supervised Instruction Tuning</a>
  * <a href='#Extract the Trained Projector'>3.3. Extract the Trained Projector</a>
  * <a href='#Task-Specific Instruction Tuning'>3.4. Task-Specific Instruction Tuning</a>
* <a href='#Evaluating GraphGPT'>4. Evaluating GraphGPT</a>
  * <a href='#Preparing Checkpoints and Data'>4.1. Preparing Checkpoints and Data</a>
  * <a href='#Running Evaluation'>4.2. Running Evaluation</a>

****



<span id='Code Structure'/>

### 1. Code Structure <a href='#all_catelogue'>[Back to Top]</a>

```
.
├── README.md
├── assets
│   ├── demo_narrow.gif
│   ├── screenshot_cli.png
│   ├── screenshot_gui.png
│   ├── server_arch.png
│   └── vicuna_logo.jpeg
├── format.sh
├── graphgpt
│   ├── __init__.py
│   ├── constants.py
│   ├── conversation.py
│   ├── eval
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── run_graphgpt.py
│   │   ├── run_graphgpt_LP.py
│   │   ├── run_vicuna.py
│   │   └── script
│   │       └── run_model_qa.yaml
│   ├── model
│   │   ├── GraphLlama.py
│   │   ├── __init__.py
│   │   ├── apply_delta.py
│   │   ├── apply_lora.py
│   │   ├── builder.py
│   │   ├── compression.py
│   │   ├── convert_fp16.py
│   │   ├── graph_layers
│   │   │   ├── __init__.py
│   │   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   │   ├── clip_graph.py
│   │   │   ├── graph_transformer.py
│   │   │   ├── mpnn.py
│   │   │   └── simple_tokenizer.py
│   │   ├── make_delta.py
│   │   ├── model_adapter.py
│   │   ├── model_registry.py
│   │   ├── monkey_patch_non_inplace.py
│   │   └── utils.py
│   ├── protocol
│   │   └── openai_api_protocol.py
│   ├── serve
│   │   ├── __init__.py
│   │   ├── api_provider.py
│   │   ├── bard_worker.py
│   │   ├── cacheflow_worker.py
│   │   ├── cli.py
│   │   ├── controller.py
│   │   ├── gateway
│   │   │   ├── README.md
│   │   │   └── nginx.conf
│   │   ├── gradio_block_arena_anony.py
│   │   ├── gradio_block_arena_named.py
│   │   ├── gradio_css.py
│   │   ├── gradio_patch.py
│   │   ├── gradio_web_server.py
│   │   ├── gradio_web_server_multi.py
│   │   ├── huggingface_api.py
│   │   ├── inference.py
│   │   ├── model_worker.py
│   │   ├── monitor
│   │   │   ├── basic_stats.py
│   │   │   ├── clean_battle_data.py
│   │   │   ├── elo_analysis.py
│   │   │   ├── hf_space_leaderboard_app.py
│   │   │   └── monitor.py
│   │   ├── openai_api_server.py
│   │   ├── register_worker.py
│   │   ├── test_message.py
│   │   └── test_throughput.py
│   ├── train
│   │   ├── graphchat_trainer.py
│   │   ├── llama_flash_attn_monkey_patch.py
│   │   ├── train_graph.py
│   │   ├── train_lora.py
│   │   └── train_mem.py
│   └── utils.py
├── playground
│   ├── inspect_conv.py
│   ├── test_embedding
│   │   ├── README.md
│   │   ├── test_classification.py
│   │   ├── test_semantic_search.py
│   │   └── test_sentence_similarity.py
│   └── test_openai_api
│       ├── anthropic_api.py
│       └── openai_api.py
├── pyproject.toml
├── scripts
│   ├── eval_script
│   │   └── graphgpt_eval.sh
│   ├── extract_graph_projector.py
│   ├── serving
│   │   ├── controller.yaml
│   │   └── model_worker.yaml
│   └── tune_script
│       ├── extract_projector.sh
│       ├── graphgpt_stage1.sh
│       └── graphgpt_stage2.sh
└── tests
    ├── test_openai_curl.sh
    ├── test_openai_langchain.py
    └── test_openai_sdk.py
```


<span id='Environment Preparation'/>






## Contact
For any questions or feedback, feel free to contact [Jingyu Xiao](mailto:jy-xiao21@mails.tsinghua.edu.cn).


## Citation

If you find SmartGuard useful in your research or applications, please kindly cite:
```tex
@articles{xiao2023smartguard,
title={}, 
author={Jingyu Xiao},
year={2024},
eprint={xxxxxxx},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```



## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat), [LLaVa](https://github.com/haotian-liu/LLaVA), We also partially draw inspirations from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). For the text-graph grounding design, we leverages implementation from [G2P2](https://github.com/WenZhihao666/G2P2). The design of our website and README.md was inspired by [NExT-GPT](https://next-gpt.github.io/). Thanks for their wonderful works.