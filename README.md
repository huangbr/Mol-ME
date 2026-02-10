# Mol-ME: Enhancing Molecular Property Prediction via Multi-Modal Self-Supervised Learning and Ensemble Methods

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of **Mol-ME**, a novel multi-modal self-supervised learning framework for molecular property prediction, as described in our paper:

> **Mol-ME: Enhancing Molecular Property Prediction via Multi-Modal Self-Supervised Learning and Ensemble Methods**  
> Baoren Huang, Mu Chen, Junjie Luo, Lei Wang, Wencai Ye, Ke Wang  

It achieves state-of-the-art performance on 9 MoleculeNet benchmark datasets.

## Quick Start
To train and evaluate Mol-ME on the BBBP dataset with seed 2022 using GPU 1:
```
python source/train.py --cfg ./configs/bbbp/bbbp.yaml --opts SEED 2022 --tag seed_2022 --gpu 1
```

## Supported Datasets
We provide config files for all 9 MoleculeNet benchmarks used in the paper:
- Classification: bace, bbbp, sider, tox21, hiv
- Regression: esol, freesolv, lipo, qm8

## Installation

We recommend using Conda to manage the environment.

```bash
# 1. Clone the repository
git clone https://github.com/huangbr/Mol-ME.git
cd Mol-ME

# 2. Create environment
conda create -n molme python=3.8
conda activate molme

# 3. Install dependencies
pip install -r requirements.txt
```

## Citation
If you use Mol-ME in your research, please cite our paper:
```
@article{huang2026molme,
  title={Mol-ME: Enhancing Molecular Property Prediction via Multi-Modal Self-Supervised Learning and Ensemble Methods},
  author={Huang, Baoren and Chen, Mu and Luo, Junjie and Wang, Lei and Ye, Wencai and Wang, Ke},
  journal={Under review},
  year={2026}
}
```

## License
This project is licensed under the MIT License
Copyright © 2026 Baoren Huang, Jinan University.
