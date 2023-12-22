# Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach

This repository contains the code for our NeurIPS 2023 accepted **Spotlight** paper, *[Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach](https://arxiv.org/abs/2310.06396)*.

## Table of Contents

- [Requirements](#requirements)
- [Reproducing Results](#reproducing-results)
- [Reference](#reference)
- [Citation](#citation)

## Requirements

To install the required dependencies, refer to the environment.yaml file

## Reproducing Results


For the non-targeted GIA in Table 2, first generate the adversarial graphs , for Cora dataset run the following command:

```bash
#pgd
python -u gnn_misg.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'pgd' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0  --use_ln 0 --grb_split
#tdgia
python -u gnn_misg.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --injection 'tdgia' --grb_split
cp atkg/cora_seqgia.pt atkg/cora_tdgia.pt

#metagia
python -u gnn_misg.py --dataset 'cora'  --inductive --eval_robo --eval_attack 'seqgia' --injection 'meta' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0  --grb_split
cp atkg/cora_seqgia.pt atkg/cora_metagia.pt

```
For other datasets in ['citeseer','pubmed','coauthorcs'], please change the value of --n_inject_max  --n_edge_max according to our paper

To evaluate the robustness of the HANG models, run the following command:

```bash
# --function: the type of HANG model, 'hang' for HANG and 'hangquad' for HANG-quad
# --eval_attack : the type of attack, 'pgd' for PGD, 'tdgia' for TDGIA, 'metagia' for METAGIA

#HANG
python gnn_misg_pde.py --dataset cora --inductive --eval_robo --eval_attack pgd --n_inject_max 60 --n_edge_max 20 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --model graphcon --time 3 --method euler --function hang --gpu 3 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split

python gnn_misg_pde.py --dataset citeseer --inductive --eval_robo --eval_attack metagia --n_inject_max 90 --n_edge_max 10 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --model graphcon --time 3 --method euler --function hang --gpu 3 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split

python gnn_misg_pde.py --dataset pubmed --inductive --eval_robo --eval_attack pgd --n_inject_max 200 --n_edge_max 100 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --model graphcon --time 3 --method euler --function hang --gpu 2 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split

python gnn_misg_pde.py --dataset coauthorcs --inductive --eval_robo --eval_attack pgd --n_inject_max 300 --n_edge_max 150 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --model graphcon --time 3 --method euler --function hang --gpu 2 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split
#HANG-quad
python gnn_misg_pde.py --dataset cora --inductive --eval_robo --eval_attack pgd --n_inject_max 60 --n_edge_max 20 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --model graphcon --time 3 --method euler --function hangquad --gpu 3 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split

python gnn_misg_pde.py --dataset citeseer --inductive --eval_robo --eval_attack metagia --n_inject_max 90 --n_edge_max 10 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --model graphcon --time 3 --method euler --function hangquad --gpu 3 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split

python gnn_misg_pde.py --dataset pubmed --inductive --eval_robo --eval_attack pgd --n_inject_max 200 --n_edge_max 100 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --model graphcon --time 3 --method euler --function hangquad --gpu 3 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split

python gnn_misg_pde.py --dataset coauthorcs --inductive --eval_robo --eval_attack pgd --n_inject_max 300 --n_edge_max 150 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --model graphcon --time 3 --method euler --function hangquad --gpu 3 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split
```

For the targeted GIA in Table 3, first generate the adversarial graphs , for Computers dataset run the following command:

```bash
#pgd
python -u gnn_misg.py --dataset 'computers'  --inductive --eval_robo --eval_attack 'pgd' --n_inject_max 100 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0  --use_ln 0 --grb_split --eval_target
#tdgia
python -u gnn_misg.py --dataset 'computers'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 100 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --injection 'tdgia' --grb_split --eval_target
cp atkg/computers_seqgia_target.pt atkg/computers_tdgia_target.pt
#metagia
python -u gnn_misg.py --dataset 'computers'  --inductive --eval_robo --eval_attack 'seqgia' --injection 'meta' --n_inject_max 100 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0  --grb_split --eval_target
cp atkg/computers_seqgia_target.pt atkg/computers_metagia_target.pt
```
For ogbn-arxiv datasets, please change the value of --n_inject_max  --n_edge_max according to our paper

To evaluate the robustness of the HANG models, run the following command:
```bash
#HANG
python gnn_misg_pde.py --dataset computers --inductive --eval_robo --eval_attack tdgia --n_inject_max 100 --n_edge_max 150 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --eval_target --eval_robo_blk --model graphcon --method euler --function hang --gpu 0 --hidden_dim 128 --step_size 1 --input_dropout 0.2 --dropout 0.4 --eval_target --batch_norm --block constant --add_source --time 3
#HANG-quad
python gnn_misg_pde.py --dataset computers --inductive --eval_robo --eval_attack tdgia --n_inject_max 100 --n_edge_max 150 --grb_mode full --runs 1 --disguise_coe 0 --use_ln 0 --eval_target --eval_robo_blk --model graphcon --method euler --function hangquad --gpu 0 --hidden_dim 128 --step_size 1 --input_dropout 0.2 --dropout 0.4 --eval_target --batch_norm --block constant --add_source --time 3

```

For the Metattack in Table 4, run the following command:
```bash

#HANG-quad
python run_metattack_rate.py --dataset polblogs --function hangquad --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 8 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 0 --epochs 800 --patience 200
#HANG
python run_metattack_rate.py --dataset polblogs --function hang --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 15 --hidden_dim 128 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 1 --epochs 800 --patience 150
#HANG
python run_metattack_rate.py --dataset pubmed --function hang --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 3 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 1 --epochs 800 --patience 150
#HANG-quad
python run_metattack_rate.py --dataset pubmed --function hangquad --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 6 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 3 --epochs 800 --patience 150


```



## Reference 

Our code is developed based on the following repos:

The GIA attack method is based on the [GIA-HAO](https://github.com/LFhase/GIA-HAO/tree/master) repo.  
The HANG model is based on the [GraphCON](https://github.com/tk-rusch/GraphCON) framework.  
The METATTACK and NETTACK methods are based on the [deeprobust](https://github.com/DSE-MSU/DeepRobust) repo.



## Citation

If you find our work useful, please cite us as follows:
```bash
@INPROCEEDINGS{ZhaKanSon:C23b,
author = {Kai Zhao and Qiyu Kang and Yang Song and Rui She and Sijie Wang and Wee Peng Tay},
title = {Adversarial Robustness in Graph Neural Networks: {A Hamiltonian} Energy Conservation Approach},
booktitle = {Advances in Neural Information Processing Systems},
volume = {},
pages = {},
month = {Dec.},
year = {2023},
address = {New Orleans, USA},
}
```



