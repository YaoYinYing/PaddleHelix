# Biomolecular Structure Prediction with HelixFold3: Replicating the Capabilities of AlphaFold3

The AlphaFold series has transformed protein structure prediction with remarkable accuracy, often matching experimental methods. While AlphaFold2 and AlphaFold-Multimer are open-sourced, facilitating rapid and reliable predictions, [AlphaFold3](https://doi.org/10.1038/s41586-024-07487-w) remains partially accessible and has not been open-sourced, restricting further development.

The PaddleHelix team is working on [HelixFold3](./helixfold3_report.pdf) to replicate the advanced capabilities of AlphaFold3. Insights from the AlphaFold3 paper inform our approach and build on our prior work with [HelixFold](https://arxiv.org/abs/2207.05477), [HelixFold-Single](https://doi.org/10.1038/s42256-023-00721-6), [HelixFold-Multimer](https://arxiv.org/abs/2404.10260), and [HelixDock](https://arxiv.org/abs/2310.13913). Currently, HelixFold3's accuracy in predicting the structures of small molecule ligands, nucleic acids (including DNA and RNA), and proteins is comparable to that of AlphaFold3. We are committed to continuously enhancing the model's performance and rigorously evaluating it across a broader range of biological molecules. Please refer to our [HelixFold3 technical report](./helixfold3_report.pdf) for more details.


<!-- <img src="./demo_output/6zcy_demo_result.png" alt="demo" align="middle" style="margin-left: 25%; margin-right: 25%; width: 50%; margin-bottom: 20px;" /> -->


<!-- <p align="center"> -->
<img src="images/ligands_posebusters_v1.png" align="left" height="60%" width="50%" style="padding-left: 10px;"/>


<img src="images/proteins_heter_v2_success_rate.png" align="right" height="60%" width="40%" style="padding-right: 10px;"/>
<br></br>

<img src="images/NA_casp15.png" style="display: block; width: 100%; padding-top: 10px;">
<br>



## HelixFold3 Inference

### Environment
Specific environment settings are required to reproduce the results reported in this repo,

* Python: 3.9
* CUDA: 12.0
* CuDNN: 8.4.0
* NCCL: 2.14.3
* Paddle: 2.6.1

Those settings are recommended as they are the same as we used in our A100 machines for all inference experiments. 

### Installation

HelixFold3 depends on [PaddlePaddle](https://github.com/paddlepaddle/paddle). Python dependencies available through `pip` 
is provided with `pyproject.toml`. `kalign`, the [`HH-suite`](https://github.com/soedinglab/hh-suite) and `jackhmmer` are 
also needed to produce multiple sequence alignments. The download scripts require `aria2c`. 

Locate to the directory of `helixfold` then run:

```bash
# Install py env
conda create -n helixfold -c conda-forge python=3.9

# activate the conda environment
conda activate helixfold

# adjust these version numbers as your situation
conda install -y cudnn=8.4.1 cudatoolkit=11.7 nccl=2.14.3 -c conda-forge -c nvidia
conda install -y -c bioconda aria2 hmmer==3.3.2 kalign2==2.04 hhsuite==3.3.0 
conda install -y -c conda-forge openbabel

# install paddlepaddle
pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# or lower version: https://paddle-wheel.bj.bcebos.com/2.5.1/linux/linux-gpu-cuda11.7-cudnn8.4.1-mkl-gcc8.2-avx/paddlepaddle_gpu-2.5.1.post117-cp39-cp39-linux_x86_64.whl

# downgrade pip
pip install --upgrade 'pip<24'

# edit configuration file at `./helixfold/config/helixfold.yaml` to set your databases and binaries correctly.

# install HF3 as a python library
pip install .  --no-cache-dir
```

Note: If you have a different version of python3 and cuda, please refer to [here](https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html) for the compatible PaddlePaddle `dev` package.


#### Install Maxit
The conversion between `.cif` and `.pdb` relies on [Maxit](https://sw-tools.rcsb.org/apps/MAXIT/index.html). 
Download Maxit source code from https://sw-tools.rcsb.org/apps/MAXIT/maxit-v11.100-prod-src.tar.gz. Untar and follow 
its `README` to complete installation. If you encouter error like your GCC version not support (9.4.0, for example), editing `etc/platform.sh` and reruning compilation again would make sense. See below:

```bash
#   Check if it is a Linux platform
    Linux)
#     Check if it is GCC version 4.x
      gcc_ver=`gcc --version | grep -e " 4\."` # edit `4\.` to `9\.`
      if [[ -z $gcc_ver ]]
```

### Usage

In order to run HelixFold3, the genetic databases and model parameters are required.

The parameters of HelixFold3 can be downloaded [here](https://paddlehelix.bd.bcebos.com/HelixFold3/params/HelixFold3-params-240814.zip), 
please place the downloaded checkpoint path in `weight_path` of `helixfold/config/helixfold.yaml` configuration file before install HF3 as a python module.

The script `scripts/download_all_data.sh` can be used to download and set up all genetic databases with the following configs:

*   By default:

    ```bash
    scripts/download_all_data.sh ./data
    ```

   will download the complete databases. The total download size for the complete databases is around 415 GB, 
   and the total size when unzipped is 2.2 TB.  

*   With `reduced_dbs`:

    ```bash
    scripts/download_all_data.sh ./data reduced_dbs
    ```

    will download a reduced version of the databases to be used with the `reduced_dbs` preset. The total download 
    size for the reduced databases is around 190 GB, and the total unzipped size is around 530 GB.

#### Understanding Model Input

There are some demo input under `./data/` for your test and reference. Data input is in the form of JSON containing
several entities such as `protein`, `ligand`, `nucleic acids`, and `iron`. Proteins and nucleic acids inputs are their sequence.
HelixFold3 supports input ligand as SMILES, CCD id or small molecule files, please refer to `/data/demo_6zcy_smiles.json` and `data/demo_p450_heme_sdf.json` 
for more details about SMILES input. Flexible input from small molecule is now supported. See `obabel -L formats |grep -v 'Write-only'`

A example of input data is as follows:

```json
{
    "entities": [
        {
            "type": "protein",
            "sequence": "MDTEVYESPYADPEEIRPKEVYLDRKLLTLEDKELGSGNFGTVKKGYYQMKKVVKTVAVKILKNEANDPALKDELLAEANVMQQLDNPYIVRMIGICEAESWMLVMEMAELGPLNKYLQQNRHVKDKNIIELVHQVSMGMKYLEESNFVHRDLAARNVLLVTQHYAKISDFGLSKALRADENYYKAQTHGKWPVKWYAPECINYYKFSSKSDVWSFGVLMWEAFSYGQKPYRGMKGSEVTAMLEKGERMGCPAGCPREMYDLMNLCWTYDVENRPGFAAVELRLRNYYYDVVNHHHHHH",
            "count": 1
        },
        {
            "type": "ligand",
            "ccd": "QF8",
            "count": 1
        }
    ]
}
```

Another example of **covalently modified** input:

```json
{
    "entities": [
        {
            "type": "protein",
            "sequence": "MDALYKSTVAKFNEVIQLDCSTEFFSIALSSIAGILLLLLLFRSKRHSSLKLPPGKLGIPFIGESFIFLRALRSNSLEQFFDERVKKFGLVFKTSLIGHPTVVLCGPAGNRLILSNEEKLVQMSWPAQFMKLMGENSVATRRGEDHIVMRSALAGFFGPGALQSYIGKMNTEIQSHINEKWKGKDEVNVLPLVRELVFNISAILFFNIYDKQEQDRLHKLLETILVGSFALPIDLPGFGFHRALQGRAKLNKIMLSLIKKRKEDLQSGSATATQDLLSVLLTFRDDKGTPLTNDEILDNFSSLLHASYDTTTSPMALIFKLLSSNPECYQKVVQEQLEILSNKEEGEEITWKDLKAMKYTWQVAQETLRMFPPVFGTFRKAITDIQYDGYTIPKGWKLLWTTYSTHPKDLYFNEPEKFMPSRFDQEGKHVAPYTFLPFGGGQRSCVGWEFSKMEILLFVHHFVKTFSSYTPVDPDEKISGDPLPPLPSKGFSIKLFPRP",
            "count": 1
        },
        {
            "type": "ligand",
            "ccd": "HEM",
            "count": 1
        },
        {
            "type": "ligand",
            "smiles": "CC1=C2CC[C@@]3(CCCC(=C)[C@H]3C[C@@H](C2(C)C)CC1)C",
            "count": 1
        },
        {
            "type": "bond",
            "bond": "A,CYS,445,SG,B,HEM,1,FE,covale,2.3",
            "_comment": "<chain-id>,<residue name>,<residue index>,<atom id>,<chain-id>,<residue name>,<residue index>,<atom id>,<bond type>,<bond length>",
            "_another_comment": "use semicolon to separate multiple bonds",
            "_also_comment": "For ccd input, use CCD key as residue name; for smiles and file input, use `UNK-<index>` where index is the chain order you input. eg. `UNK-1` for the first ligand chain(or the count #1), `UNK-2` the second(or the count #2)."
        }
    ]
}
```

For seaking all atom ids in CCD database:

```shell
helixfold_show_ccd +ccd_id=HEM
```

This command outputs like:

```text
# output:
[2024-08-23 22:44:36,324][absl][INFO] - Started Loading CCD dataset from /mnt/db/ccd/ccd_preprocessed_etkdg.pkl.gz
[2024-08-23 22:44:43,236][absl][INFO] - Finished Loading CCD dataset from /mnt/db/ccd/ccd_preprocessed_etkdg.pkl.gz in 6.912 seconds
[2024-08-23 22:44:43,237][absl][INFO] - CCD dataset contains 43488 entries.
[2024-08-23 22:44:43,237][absl][INFO] - Atoms in HEM: ['CHA', 'CHB', 'CHC', 'CHD', 'C1A', 'C2A', 'C3A', 'C4A', 'CMA', 'CAA', 'CBA', 'CGA', 'O1A', 'O2A', 'C1B', 'C2B', 'C3B', 'C4B', 'CMB', 'CAB', 'CBB', 'C1C', 'C2C', 'C3C', 'C4C', 'CMC', 'CAC', 'CBC', 'C1D', 'C2D', 'C3D', 'C4D', 'CMD', 'CAD', 'CBD', 'CGD', 'O1D', 'O2D', 'NA', 'NB', 'NC', 'ND', 'FE']
```

For seaking all atom ids in a given `sdf`/`mol2`, the atom list follows the same order in its file.

HF3 parsed:

```text
['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'N1', 'O1', 'O2', 'O3', 'O4', 'O5', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'N2', 'O6', 'O7', 'O8', 'O9', 'O10', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'O11', 'O12', 'O13', 'O14', 'O15', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'O16', 'O17', 'O18', 'O19', 'O20', 'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'O21', 'O22', 'O23', 'O24', 'O25', 'C35', 'C36', 'C37', 'C38', 'C39', 'C40', 'O26', 'O27', 'O28', 'O29', 'O30']
```

while in `SDF`:

```text
   29.7340    3.2540   76.7430 C   0  0  0  0  0  2  0  0  0  0  0  0
   29.8160    4.4760   77.6460 C   0  0  1  0  0  3  0  0  0  0  0  0
   28.5260    5.2840   77.5530 C   0  0  2  0  0  3  0  0  0  0  0  0
   28.1780    5.5830   76.1020 C   0  0  1  0  0  3  0  0  0  0  0  0
   28.2350    4.3240   75.2420 C   0  0  1  0  0  3  0  0  0  0  0  0
   28.1040    4.6170   73.7650 C   0  0  0  0  0  2  0  0  0  0  0  0
   31.3020    3.8250   79.4830 C   0  0  0  0  0  0  0  0  0  0  0  0
   31.3910    3.4410   80.9280 C   0  0  0  0  0  1  0  0  0  0  0  0
   30.0760    4.0880   79.0210 N   0  0  0  0  0  2  0  0  0  0  0  0
   28.6870    6.5050   78.2670 O   0  0  0  0  0  1  0  0  0  0  0  0
   26.8490    6.0910   76.0350 O   0  0  0  0  0  0  0  0  0  0  0  0
   29.4950    3.6650   75.4130 O   0  0  0  0  0  0  0  0  0  0  0  0
   29.3670    4.5550   73.1150 O   0  0  0  0  0  1  0  0  0  0  0  0
   32.2950    3.8940   78.7640 O   0  0  0  0  0  0  0  0  0  0  0  0
   26.7420    7.4140   75.6950 C   0  0  1  0  0  3  0  0  0  0  0  0
   25.2700    7.7830   75.6110 C   0  0  1  0  0  3  0  0  0  0  0  0
   25.1290    9.2300   75.1610 C   0  0  2  0  0  3  0  0  0  0  0  0
   25.9180   10.1440   76.0880 C   0  0  1  0  0  3  0  0  0  0  0  0
   27.3630    9.6720   76.2210 C   0  0  1  0  0  3  0  0  0  0  0  0
   28.1310   10.4360   77.2730 C   0  0  0  0  0  2  0  0  0  0  0  0
   23.8820    5.8170   75.1400 C   0  0  0  0  0  0  0  0  0  0  0  0
   23.1980    5.0100   74.0810 C   0  0  0  0  0  1  0  0  0  0  0  0
   24.5530    6.8930   74.7160 N   0  0  0  0  0  2  0  0  0  0  0  0
   23.7530    9.5950   75.1670 O   0  0  0  0  0  1  0  0  0  0  0  0
   25.9170   11.4700   75.5730 O   0  0  0  0  0  0  0  0  0  0  0  0
   27.4050    8.2900   76.6040 O   0  0  0  0  0  0  0  0  0  0  0  0
   29.5300   10.4030   77.0280 O   0  0  0  0  0  1  0  0  0  0  0  0
   23.8300    5.5110   76.3290 O   0  0  0  0  0  0  0  0  0  0  0  0
   25.3940   12.4250   76.4090 C   0  0  1  0  0  3  0  0  0  0  0  0
   25.9490   13.7680   75.9090 C   0  0  2  0  0  3  0  0  0  0  0  0
   25.1320   14.9560   76.4900 C   0  0  2  0  0  3  0  0  0  0  0  0
   23.6130   14.6900   76.6390 C   0  0  1  0  0  3  0  0  0  0  0  0
   23.3700   13.3000   77.2280 C   0  0  1  0  0  3  0  0  0  0  0  0
   21.9020   12.9360   77.3500 C   0  0  0  0  0  2  0  0  0  0  0  0
   25.9010   13.8490   74.4810 O   0  0  0  0  0  1  0  0  0  0  0  0
   25.3420   16.1410   75.7110 O   0  0  0  0  0  0  0  0  0  0  0  0
   23.0420   15.6520   77.5170 O   0  0  0  0  0  1  0  0  0  0  0  0
   23.9910   12.3690   76.3570 O   0  0  0  0  0  0  0  0  0  0  0  0
   21.3660   12.8480   76.0500 O   0  0  0  0  0  0  0  0  0  0  0  0
   20.8090   11.6500   75.6780 C   0  0  2  0  0  3  0  0  0  0  0  0
   20.6800   11.6410   74.1740 C   0  0  2  0  0  3  0  0  0  0  0  0
   19.5510   12.5850   73.8180 C   0  0  2  0  0  3  0  0  0  0  0  0
   18.2370   12.0940   74.4540 C   0  0  1  0  0  3  0  0  0  0  0  0
   18.4030   11.9240   75.9810 C   0  0  1  0  0  3  0  0  0  0  0  0
   17.2710   11.1260   76.6120 C   0  0  0  0  0  2  0  0  0  0  0  0
   20.2900   10.3510   73.7080 O   0  0  0  0  0  1  0  0  0  0  0  0
   19.4280   12.7380   72.4110 O   0  0  0  0  0  0  0  0  0  0  0  0
   17.2120   13.0460   74.2030 O   0  0  0  0  0  1  0  0  0  0  0  0
   19.6260   11.2000   76.3010 O   0  0  0  0  0  0  0  0  0  0  0  0
   16.0670   11.4490   75.9360 O   0  0  0  0  0  1  0  0  0  0  0  0
   20.2190   13.6280   71.7260 C   0  0  2  0  0  3  0  0  0  0  0  0
   19.6090   14.0000   70.3810 C   0  0  2  0  0  3  0  0  0  0  0  0
   19.6360   12.7820   69.4880 C   0  0  2  0  0  3  0  0  0  0  0  0
   21.0860   12.3100   69.3240 C   0  0  1  0  0  3  0  0  0  0  0  0
   21.7030   12.0240   70.7120 C   0  0  1  0  0  3  0  0  0  0  0  0
   23.1940   11.7460   70.6620 C   0  0  0  0  0  2  0  0  0  0  0  0
   20.4080   14.9810   69.7000 O   0  0  0  0  0  1  0  0  0  0  0  0
   19.0310   13.0500   68.2340 O   0  0  0  0  0  1  0  0  0  0  0  0
   21.1060   11.1280   68.5380 O   0  0  0  0  0  1  0  0  0  0  0  0
   21.5380   13.1700   71.5840 O   0  0  0  0  0  0  0  0  0  0  0  0
   23.8240   12.5210   71.6820 O   0  0  0  0  0  1  0  0  0  0  0  0
   26.0070   17.3020   76.0200 C   0  0  2  0  0  3  0  0  0  0  0  0
   27.0750   17.5250   74.9350 C   0  0  2  0  0  3  0  0  0  0  0  0
   28.3660   16.8320   75.3290 C   0  0  2  0  0  3  0  0  0  0  0  0
   28.7820   17.2470   76.7510 C   0  0  1  0  0  3  0  0  0  0  0  0
   27.6930   16.8120   77.7320 C   0  0  1  0  0  3  0  0  0  0  0  0
   27.9770   17.2020   79.1710 C   0  0  0  0  0  2  0  0  0  0  0  0
   27.3990   18.9140   74.8010 O   0  0  0  0  0  1  0  0  0  0  0  0
   29.4060   17.0990   74.3950 O   0  0  0  0  0  1  0  0  0  0  0  0
   30.0160   16.6410   77.0930 O   0  0  0  0  0  1  0  0  0  0  0  0
   26.4610   17.4820   77.3520 O   0  0  0  0  0  0  0  0  0  0  0  0
   27.3660   18.4620   79.4040 O   0  0  0  0  0  1  0  0  0  0  0  0
```

#### Running HelixFold for Inference

To run inference on a sequence or multiple sequences using HelixFold3's pretrained parameters, run e.g.:

##### Run from default config

```shell
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH \
helixfold \
    input=./data/demo_8ecx.json \
    output=. \
    CONFIG_DIFFS.preset=allatom_demo
```

##### Run with customized configuration dir and file(`./myfold.yaml`, for example):

```shell
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH \
helixfold --config-dir=. --config-name=myfold \
    input=./data/demo_6zcy_smiles.json \
    output=. \
    CONFIG_DIFFS.preset=allatom_demo
```

##### Run with additional configuration term 

```shell
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH \
helixfold \
    input=./data/demo_6zcy.json \
    output=. \
    CONFIG_DIFFS.preset=allatom_demo \
    +CONFIG_DIFFS.model.global_config.subbatch_size=192 \
    +CONFIG_DIFFS.model.num_recycle=10
```

The descriptions of the above script are as follows:
* `LD_LIBRARY_PATH` - This is required to load the `libcudnn.so` library if you encounter issue like `RuntimeError: (PreconditionNotMet) Cannot load cudnn shared library. Cannot invoke method cudnnGetVersion.`
* `config-dir` - The directory that contains the alterative configuration file you would like to use.
* `config-name` - The name of the configuration file you would like to use.
* `input` - Input data in the form of JSON. Input pattern in `./data/demo_*.json` for your reference.
* `output` - Model output path. The output will be in a folder named the same as your `--input_json` under this path.
* `CONFIG_DIFFS.preset` - Adjusted model config preset name in `./helixfold/model/config.py:CONFIG_DIFFS`. The preset will be updated into final model configuration with `CONFIG_ALLATOM`.
* `CONFIG_DIFFS.*` - Override model any configuration in `CONFIG_ALLATOM`.

### Understanding Model Output

The outputs will be in a subfolder of `output_dir`, including the computed MSAs, predicted structures, 
ranked structures, and evaluation metrics. For a task of inferring twice with diffusion batch size 3, 
assume your input JSON is named `demo_data.json`, the `output_dir` directory will have the following structure:

```text
<output_dir>/
└── demo_data/
    ├── demo_data-pred-1-1/
    │   ├── all_results.json
    │   ├── predicted_structure.pdb
    │   └── predicted_structure.cif
    ├── demo_data-pred-1-2/
    ├── demo_data-pred-1-3/
    ├── demo_data-pred-2-1/
    ├── demo_data-pred-2-2/
    ├── demo_data-pred-2-3/
    |
    ├── demo_data-rank[1-6]/
    │   ├── all_results.json
    |   ├── predicted_structure.pdb
    │   └── predicted_structure.cif  
    |
    ├── final_features.pkl
    └── msas/
        ├── ...
        └── ...

```

The contents of each output file are as follows:
* `final_features.pkl` – A `pickle` file containing the input feature NumPy arrays
 used by the models to predict the structures. If you need to re-run a inference without re-building the MSAs, delete this file.
* `msas/` - A directory containing the files describing the various genetic
 tool hits that were used to construct the input MSA.
* `demo_data-pred-X-Y` - Prediction results of `demo_data.json` in X-th inference and Y-thdiffusion batch, 
including predicted structures in `cif` or `pdb` and a JSON file containing all metrics' results.
* `demo_data-rank*` - Ranked results of a series of predictions according to metrics.

### Resource Usage

We suggest a single GPU for inference has at least 32G available memory. The maximum number of tokens is around 
1200 for inference on a single A100-40G GPU with precision `bf16`. The length of inference input tokens on a 
single V100-32G with precision `fp32` is up to 1000. Inferring longer tokens or entities with larger atom numbers 
per token than normal protein residues like nucleic acids may cost more GPU memory.

For samples with larger tokens, you can override `model.global_config.subbatch_size` in `CONFIG_ALLATOM` by using `+CONFIG_DIFFS.model.global_config.subbatch_size=X` on command runs, where `X` is a smaller number than `96`, to save more GPU memory although this will cause a slower inference. Additionally, you can reduce the number of additional recycles by setting `+CONFIG_DIFFS.model.num_recycle=Y`, where `Y` is a smaller number than `3`.


We are keen on support longer token inference, it will come in soon.


## Copyright

HelixFold3's code and model parameters are available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/),  license for non-commercial use by individuals or non-commercial organizations only. Please check the details in [LICENSE](./LICENSE) before using HelixFold3.

## Reference

[1]  Abramson, J et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 630, 493–500. 10.1038/s41586-024-07487-w

[2] Jumper J, Evans R, Pritzel A, et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature 577 (7792), 583–589. 10.1038/s41586-021-03819-2.

[3] Evans, R. et al. (2022). Protein complex prediction with AlphaFold-Multimer. Preprint at bioRxiv https://doi.org/10.1101/2021.10.04.463034

[4]  Guoxia Wang, Xiaomin Fang, Zhihua Wu, Yiqun Liu, Yang Xue, Yingfei Xiang, Dianhai Yu, Fan Wang,
and Yanjun Ma. Helixfold: An efficient implementation of alphafold2 using paddlepaddle. arXiv preprint
arXiv:2207.05477, 2022

[5] Xiaomin Fang, Fan Wang, Lihang Liu, Jingzhou He, Dayong Lin, Yingfei Xiang, Kunrui Zhu, Xiaonan Zhang,
Hua Wu, Hui Li, et al. A method for multiple-sequence-alignment-free protein structure prediction using a protein
language model. Nature Machine Intelligence, 5(10):1087–1096, 2023

[6] Xiaomin Fang, Jie Gao, Jing Hu, Lihang Liu, Yang Xue, Xiaonan Zhang, and Kunrui Zhu. Helixfold-multimer:
Elevating protein complex structure prediction to new heights. arXiv preprint arXiv:2404.10260, 2024.

[7] Lihang Liu, Donglong He, Xianbin Ye, Shanzhuo Zhang, Xiaonan Zhang, Jingbo Zhou, Jun Li, Hua Chai, Fan
Wang, Jingzhou He, et al. Pre-training on large-scale generated docking conformations with helixdock to unlock
the potential of protein-ligand structure prediction models. arXiv preprint arXiv:2310.13913, 2023.

## Citation

If you use the code, data, or checkpoints in this repo, please cite the following:

```bibtex
@article{helixfold3,
  title={Technical Report of HelixFold3 for Biomolecular Structure Prediction},
  author={PaddleHelix Team},
  year={2024}
}
```
