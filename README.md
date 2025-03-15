<div align="center">

# MMSafe-PO
**Towards Harmless Multimodal Assistants with Preference Optimization**

<a href='https://huggingface.co/datasets/Downton/MMSafe-PO'><img src='https://img.shields.io/badge/Huggingface-Dataset-FFCC33'></a>
<a href='https://lu-yang666.github.io/MMsafe-PO-Web/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
</div>


## Brief Introduction

This repository contains the code and data of **MMSafe-PO**.

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in multimodal understanding, reasoning, and interaction. Given the extensive applications of MLLMs, the associated safety issues have become increasingly critical. Due to the effectiveness of preference optimization in aligning MLLMs with human preferences, there is an urgent need for safety-related preference data for MLLMs. To address this, we construct the MMSafe-PO preference dataset towards harmless multimodal assistants, featuring multimodal instructions, the conversational format, and ranked paired responses from human feedback. We also identify two insightful observations: modality co-defense and modality cheating, which illustrate that MLLMs possess a certain level of inherent defense while still presenting unique safety challenges. Based on these observations, we propose the Blind Preference Optimization (BPO) approach. Comprehensive experiments on three benchmarks show that BPO effectively enhances the safety capabilities of MLLMs. Notably, BPO significantly improves the safety rate of the base MLLM by 45.0\%, outperforming the DPO approach. Additionally, applying BPO to the MMSafe-PO dataset greatly reduces the base MLLM's unsafe rate on other safety benchmarks (14.5\% on MM-SafetyBench and 82.9\% on HarmEval, demonstrating the effectiveness and robustness of both the dataset and the approach.


## Contents 

- [Dataset](#dataset)
- [Ckpts](#ckpts)
- [BPO Training](#bpo-training)
- [Infer](#infer)

## Dataset

We present the **MMSafe-PO** Dataset, featuring multimodal instructions, the conversational format, and ranked paired responses from human feedback. The dataset can be downloaded from `datasets` folder.

## Ckpts
We put all our checkpoints into [MMSafe_checkpoints](https://huggingface.co/Downton/MMSafe_checkpoints).

## BPO Training

1. Prepare training environment

First of all, you should download [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) codes, then download our codes, put our code folder into llava folder in LLaVA-v1.5.

```bash
# Download LLaVA-v1.5
git clone https://github.com/haotian-liu/LLaVA.git

# Creating conda environment
conda create -n mmsafe python=3.10 -y
conda activate mmsafe
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install chardet

# Installing dependencies
pip install -e . && pip install datasets tensorboard deepspeed

# Download BPO training codes
cd llava
git clone THIS_GIT
```
2. Download llava-v1.5 checkpoint

Before RLHF training, you should download LLaVA-v1.5-7b checkpoint from [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-7b); LLaVA-v1.5-13b checkpoint from(https://huggingface.co/liuhaotian/llava-v1.5-13b).

3. Prepare the dataset
```
Download our dataset, generate corresponding responses using LLaVA, and use the generated responses as new rejected data. Then, concatenate them with the original data to form the training data for BPO.
Also, prepare a JSON file that includes the paths to all the JSON files you want to use for training. The format should be as follows:
```json
{
    "keyword1": "/path/to/your/dataset1.json",
    "keyword2": "/path/to/your/dataset2.json",
    "keyword3": "/path/to/your/dataset3.json"
}

```

4. MMSafe-PO BPO Training

You should start by completing the paths as required. And then run the following script.

```bash
bash ./scripts/bpo.sh
```

## Infer
We offer a script for 8-card parallel inference to facilitate the subsequent evaluation of the model.
You can run the following script.

```bash
bash ./scripts/infer_json.sh
```

