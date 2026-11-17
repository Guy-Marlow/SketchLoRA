# SketchLoRA

- This repository contains our implementation for Sketched Adapters for Continual Learning. 
- SketchLoRA approximates growing banks of per-task LoRA adapters with a fixed-rank sketch.

## Setup

Clone the repository:

```
git clone <repo-url>
cd SketchLoRA
```

Construct the environment by running:

```
conda create -n <name> python=3.11 -y
conda activate <name>
```

Install the required libraries by running:

```
pip install -r requirements.txt
```

Download each dataset into /data by running:

```
python scripts/data_prep.py --dataset <dataset>
```

Where dataset is any of [cifar100, imagenetr, food101, sun397, omnibenchmark1k].

## Training and Evaluation

You can reproduce our main-table experiments by running:

```
python scripts/run_baselines.py
python scripts/run_sketch_baselines.py
```

## Acknowledgements

This repository is built on [PILOT](https://github.com/sun-hailong/LAMDA-PILOT), with reference code from the official repositories of [InfLoRA](https://github.com/liangyanshuo/InfLoRA), [SD-LoRA](https://github.com/WuYichen-97/SD-Lora-CL), [O-LoRA](https://github.com/cmnfriend/O-LoRA) and [RainbowPrompt](https://github.com/Kiseong0753/RainbowPrompt). We thank the authors for their research, and for making their code publicly available.