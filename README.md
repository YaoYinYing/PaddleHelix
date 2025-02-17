English | [简体中文](README_cn.md)

<p align="center">
<img src="./.github/paddlehelix_logo.png" align="middle" height="90%" width="90%" />
</p>

------
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHelix.svg)](https://github.com/PaddlePaddle/PaddleHelix/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
[![DOI](https://zenodo.org/badge/314704349.svg)](https://zenodo.org/badge/latestdoi/314704349)


## Latest News
`2024.08.15` PaddleHelix released the codes and model parameters of HelixFold3, biomolecular structure prediction replicating the capabilities of AlphaFold3. HelixFold3 achieves accuracy comparable to AlphaFold3 in predicting the structures of the conventional ligands, nucleic acids, and proteins. The initial release of HelixFold3 is available as open source on GitHub for non-commercial academic research, promising to advance biomolecular research and accelerate discoveries. Refer to [codes](./apps/protein_folding/helixfold3) for more details. 

`2024.05.23` PaddleHelix released the codes of HelixDock, a pre-training model on large-scale generated docking conformations to unlock the potential of protein-ligand structure prediction, significantly improving prediction accuracy and generalizability. Please refer to [paper]([https://arxiv.org/abs/2310.13913) and [codes](./apps/molecular_docking/helixdock) for more details. Welcome to [PaddleHelix website](https://paddlehelix.baidu.com/app/drug/helix-dock/forecast) to try out the structure prediction online service. 

`2024.05.13` Paper "Multi-purpose RNA Language Modeling with Motif-aware Pre-training and Type-guided Fine-tuning" is accepted by Nature Machine Intelligence. Please refer to [paper](https://www.nature.com/articles/s42256-024-00836-4) and [codes](https://github.com/CatIIIIIIII/RNAErnie) for more details.


`2024.04.16` PaddleHelix released the technical report of HelixFold-Multimer, a protein complex structure prediction model which achieves remarkable success in antigen-antibody and peptide-protein structure prediction. Please refer to the [report](https://arxiv.org/abs/2404.10260v2) for more details. The online structure prediction services for general and antigen-antibody protein complex are now available at [link1](https://paddlehelix.baidu.com/app/drug/protein-complex/forecast) and [link2](https://paddlehelix.baidu.com/app/drug/KYKT/forecast) on the PaddleHelix platform respectively.

`2023.10.09` The work of HelixFold-Single titled with "A method for multiple-sequence-alignment-free protein structure prediction using a protein language model" is accepted by Nature Machine Intelligence. Please refer to [paper](https://doi.org/10.1038/s42256-023-00721-6) for more details.


`2022.12.08` Paper "HelixMO: Sample-Efficient Molecular Optimization in Scene-Sensitive Latent Space" is accepted by **BIBM 2022**. Please refere to [link1](https://www.computer.org/csdl/proceedings-article/bibm/2022/09995561/1JC23yWxizC) or [link2](https://aps.arxiv.org/abs/2112.00905) for more details. We also deployed the drug design service on the website [PaddleHelix](https://paddlehelix.baidu.com/app/drug/drugdesign/forecast).

`2022.08.11` PaddleHelix released the codes of HelixGEM-2, a novel Molecular Property Prediction Network that models full-range many-body interactions. And it ranked 1st in the OGB [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/leaderboards/) leaderboard. Please refer to [paper](https://arxiv.org/abs/2208.05863) and [codes](./apps/pretrained_compound/ChemRL/GEM-2) for more details.

`2022.07.29` PaddleHelix released the codes of HelixFold-Single, an **MSA-free** protein structure prediction pipeline relying on only the primary sequences, which can **predict the protein structures within seconds**. Please refer to [paper](https://arxiv.org/abs/2207.13921) and [codes](./apps/protein_folding/helixfold-single) for more details. Welcome to [PaddleHelix website](https://paddlehelix.baidu.com/app/drug/protein-single/forecast) to try out the structure prediction online service.

`2022.07.18` PaddleHelix fully released HelixFold including training and inference pipeline. **The complete training time are optimized from 11 days to 5.12 days. Ultra-long monomer protein (around 6600 AA) prediction is supported now**. Please refer to [paper](https://arxiv.org/abs/2207.05477) and [codes](./apps/protein_folding/helixfold) for more details.

`2022.07.07` Paper "BatchDTA: implicit batch alignment enhances deep learning-based drug–target affinity estimation" is published in **Briefings in Bioinformatics**. Please refer to [paper](https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbac260/6632927) and [codes](./apps/drug_target_interaction/batchdta) for more details.

`2022.05.24` Paper "HelixADMET: a robust and endpoint extensible ADMET system incorporating self-supervised knowledge transfer" is published in **Bioinformatics**. Refer to [paper](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btac342/6590643) for more information.

`2022.02.07` Paper "Geometry-enhanced molecular representation learning for property prediction" is published in **Nature Machine Intelligence**. Please refer to [paper](https://www.nature.com/articles/s42256-021-00438-4) and [codes](./apps/pretrained_compound/ChemRL/GEM) to explore the algorithm.

<details>
<summary>More news ...</summary>

`2022.01.07` PaddleHelix released the reproduction of [AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2) inference pipeline using PaddlePaddle in [HelixFold](./apps/protein_folding/helixfold).

`2021.11.23` Paper "Multimodal Pre-Training Model for Sequence-based Prediction of Protein-Protein Interaction" is accepted by [MLCB 2021](https://sites.google.com/cs.washington.edu/mlcb2021/home). Please refer to [paper](https://arxiv.org/abs/2112.04814) and [code](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_protein_interaction) for more details.

`2021.10.25` Paper "Docking-based Virtual Screening with Multi-Task Learning" is accepted by [BIBM 2021](https://ieeebibm.org/BIBM2021/).

`2021.09.29` Paper "Property-Aware Relation Networks for Few-shot Molecular Property Prediction" is accepted by [NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/91bc333f6967019ac47b49ca0f2fa757-Abstract.html) as a Spotlight Paper. Please refer to [PAR](./apps/fewshot_molecular_property) for more details.

`2021.07.29` PaddleHelix released a novel geometry-level molecular pre-training model, taking advantage of the 3D spatial structures of the molecules. Please refer to [GEM](./apps/pretrained_compound/ChemRL/GEM) for more details.

`2021.06.17` PaddleHelix team won the 2nd place in the [OGB-LCS KDD Cup 2021 PCQM4M-LSC track](https://ogb.stanford.edu/kddcup2021/results/), predicting DFT-calculated HOMO-LUMO energy gap of molecules. Please refer to [the solution](./competition/kddcup2021-PCQM4M-LSC) for more details.

`2021.05.20` PaddleHelix v1.0 released. 1) Update from static framework to dynamic framework; 2) Add new applications: molecular generation and drug-drug synergy.

`2021.05.18` Paper "Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity" is accepted by [KDD 2021](https://kdd.org/kdd2021/accepted-papers/index). The code is available at [here](./apps/drug_target_interaction/sign).

`2021.03.15` PaddleHelix team ranks 1st in the ogbg-molhiv and ogbg-molpcba of [OGB](https://ogb.stanford.edu/docs/leader_graphprop/), predicting the molecular properties.
</details>

------

## Introduction
PaddleHelix is a bio-computing tool, taking advantage of the machine learning approaches, especially deep neural networks, for facilitating the development of the following areas:
* **Drug Discovery**. Provide 1) Large-scale pre-training models: compounds and proteins; 2) Various applications: molecular property prediction, drug-target affinity prediction, and molecular generation.
* **Vaccine Design**. Provide RNA design algorithms, including LinearFold and LinearPartition.
* **Precision Medicine**. Provide application of drug-drug synergy.

<p align="center">
<img src=".github/PaddleHelix_Structure.png" align="middle" heigh="70%" width="70%" />
</p>

## Resources
### Application Platform
**[PaddleHelix platform](https://paddlehelix.baidu.com/)** provides the AI + biochemistry abilities for the scenarios of drug discovery, vaccine design and precision medicine.

### Installation Guide
PaddleHelix is a bio-computing repository based on [PaddlePaddle](https://github.com/paddlepaddle/paddle), a high-performance Parallelized Deep Learning Platform. The installation prerequisites and guide can be found [here](./installation_guide.md).

### Tutorials
We provide abundant [tutorials](./tutorials) to help you navigate the repository and start quickly.
* **Drug Discovery**
  - [Compound Representation Learning and Property Prediction](./tutorials/compound_property_prediction_tutorial.ipynb)
  - [Protein Representation Learning and Property Prediction](./tutorials/protein_pretrain_and_property_prediction_tutorial.ipynb)
  - [Predicting Drug-Target Interaction: GraphDTA](./tutorials/drug_target_interaction_graphdta_tutorial.ipynb), [MolTrans](./tutorials/drug_target_interaction_moltrans_tutorial.ipynb)
  - [Molecular Generation](./tutorials/molecular_generation_tutorial.ipynb)
* **Vaccine Design**
  - [Predicting RNA Secondary Structure](./tutorials/linearrna_tutorial.ipynb)

### Examples
We also provide [examples](./apps) that implement various algorithms and show the methods running the algorithms:
* **Pretraining**
  - [Representation Learning - Compounds](./apps/pretrained_compound)
  - [Representation Learning - Proteins](./apps/pretrained_protein)
* **Drug Discovery and Precision Medicine**
  - [Drug-Target Interaction](./apps/drug_target_interaction)
  - [Molecular Generation](./apps/molecular_generation)
  - [Drug Drug Synergy](./apps/drug_drug_synergy)
  - [Few-shot Molecular Property Prediction](./apps/fewshot_molecular_property)
* **Vaccine Design**
  - [LinearRNA](./c/pahelix/toolkit/linear_rna)
* **Protein Structure Prediction**
  - [HelixFold](./apps/protein_folding/helixfold)
  - [HelixFold-Single](./apps/protein_folding/helixfold-single)
  - [HelixFold3](./apps/protein_folding/helixfold3)

### Competition Solutions
The PaddleHelix team participated in multiple competitions related to bio-computing. The solutions can be found [here](./competition).

### Guide for Developers
* To develop new functions based on the source code of PaddleHelix, please refer to [guide for developers](./developer_guide.md).
* For more details on the APIs, please refer to the [documents](https://paddlehelix.readthedocs.io/en/dev/).

------

## Copyright and License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
