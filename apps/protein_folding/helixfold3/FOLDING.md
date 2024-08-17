# FOLD GUIDE

## Quick example
Run from default config
```shell
LD_LIBRARY_PATH=/mnt/data/envs/conda_env/envs/helixfold/lib/:$LD_LIBRARY_PATH helixfold input=/repo/PaddleHelix/apps/protein_folding/helixfold3/data/demo_8ecx.json  output=. CONFIG_DIFFS.preset=allatom_demo
```

Run with customized configuration dir and name:
```shell
LD_LIBRARY_PATH=/mnt/data/envs/conda_env/envs/helixfold/lib/:$LD_LIBRARY_PATH helixfold --config-dir=. --config-name=myfold input=/repo/PaddleHelix/apps/protein_folding/helixfold3/data/demo_6zcy_smiles.json  output=. CONFIG_DIFFS.preset=allatom_demo
```