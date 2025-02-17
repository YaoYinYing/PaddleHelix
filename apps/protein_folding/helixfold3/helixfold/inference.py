#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference scripts."""
from dataclasses import dataclass
import re
import os
import copy
import random
import ml_collections
import paddle
import json
import pickle
import pathlib
import shutil
import numpy as np
import shutil

from absl import logging

from omegaconf import DictConfig
import hydra

from helixfold.common import all_atom_pdb_save
from helixfold.data.pipeline_conf_bonds import load_ccd_dict
from helixfold.model import config, utils
from helixfold.data import pipeline_parallel as pipeline
from helixfold.data import pipeline_multimer_parallel as pipeline_multimer
from helixfold.data import pipeline_rna_parallel as pipeline_rna
from helixfold.data import pipeline_rna_multimer
from helixfold.data.utils import atom_level_keys, map_to_continuous_indices
from helixfold.utils.model import RunModel
from helixfold.data.tools import hmmsearch
from helixfold.data import templates
from helixfold.model.config import CONFIG_ALLATOM, CONFIG_DIFFS
from helixfold.data.tools.utils import timing
from helixfold.utils.utils import get_custom_amp_list
from typing import Any, Dict, Mapping, Union
from helixfold.utils import feature_processing_aa, preprocess
from helixfold.utils import mmcif_writer


script_path=os.path.dirname(__file__)

ALLOWED_LIGAND_BONDS_TYPE_MAP = preprocess.ALLOWED_LIGAND_BONDS_TYPE_MAP
INVERSE_ALLOWED_LIGAND_BONDS_TYPE_MAP = {
    v: k for k, v in ALLOWED_LIGAND_BONDS_TYPE_MAP.items()
}

DISPLAY_RESULTS_KEYS = [
    'atom_chain_ids',
    'atom_plddts',
    'pae',
    'token_chain_ids',
    'token_res_ids',
    'iptm',
    'ptm',
    'ranking_confidence',
    'has_clash', 
    'mean_plddt',
]

RETURN_KEYS = ['diffusion_module', 'confidence_head']


MAX_TEMPLATE_HITS = 4

def init_seed(seed):
    """ set seed for reproduct results"""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def batch_convert(np_array, add_batch=True):
    np_type = {}
    other_type = {}
    # 
    for key, value in np_array.items():
        if type(value) == np.ndarray:
            np_type.update(utils.map_to_tensor({key: value}, add_batch=add_batch))
        else:
            other_type[key] = [value]  ## other type shoule be list.
    
    return {**np_type, **other_type}



def convert_to_json_compatible(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_compatible(i) for i in obj]
    return obj

def resolve_bin_path(cfg_path: str, default_binary_name: str)-> str:
    """Helper function to resolve the binary path."""
    if cfg_path and os.path.isfile(cfg_path):
        return cfg_path

    if cfg_val:=shutil.which(default_binary_name):
        logging.warning(f'Using resolved {default_binary_name}: {cfg_val}')
        return cfg_val

    raise FileNotFoundError(f"Could not find a proper binary path for {default_binary_name}: {cfg_path}.")


# inference only, with hydra and omegaconf
def update_model_config(config_diffs: Union[str, DictConfig, ml_collections.ConfigDict, Mapping[str, dict[str, Any]]]) -> DictConfig:
  """Get the ConfigDict of a model."""

  if hasattr(config.CONFIG_ALLATOM, 'to_dict'): # ml_collections.ConfigDict to DictConfig
    cfg_aa=CONFIG_ALLATOM.to_dict()
  else:
    cfg_aa=CONFIG_ALLATOM

  cfg = copy.deepcopy(DictConfig(cfg_aa))
  if config_diffs is None or config_diffs=='':
    # early return if nothing is changed
    return cfg

  if isinstance(config_diffs, DictConfig):
    if 'preset' in config_diffs and (preset_name:=config_diffs['preset']) in CONFIG_DIFFS:
      CONFIG_DIFFS_DOTLIST={k:[f'{i}={j}' for i, j in v.items()] for k,v in CONFIG_DIFFS.items()}
      updated_config=CONFIG_DIFFS_DOTLIST[preset_name]
      cfg.merge_with_dotlist(updated_config)
      print(f'Updated config from `CONFIG_DIFFS.{preset_name}`: {updated_config}')

    # update from detailed configuration
    if any(root_kw in config_diffs for root_kw in CONFIG_ALLATOM):

      for root_kw in CONFIG_ALLATOM:
        if root_kw not in config_diffs:
          continue
        cfg.merge_with(DictConfig({root_kw:config_diffs[root_kw]})) # merge to override
        print(f'Updated config from `CONFIG_DIFFS`:{root_kw}: {config_diffs[root_kw]}')
    
    return cfg
  
  raise ValueError(f'Invalid config_diffs ({type(config_diffs)}): {config_diffs}')
    

def load_to_dev_shm(file_path: str, ramdisk_path: str = "/dev/shm", keep:bool=False) -> str:
    """
    Copies a file to /dev/shm (RAM-backed filesystem) and returns the path.
    
    :param file_path: The path to the large file on the disk.
    :return: The path to the file in /dev/shm.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Ensure the RAM disk path exists and is a directory
    if not os.path.isdir(ramdisk_path):
        raise NotADirectoryError(f"RAM disk path not found or not a directory: {ramdisk_path}")

    

    target_path = os.path.join(ramdisk_path, pathlib.Path(file_path).name)

    if os.path.isfile(target_path) and keep:
        logging.info(f"File already exists in RAM disk: {target_path}")
        return target_path

    with timing(f'loading {file_path} -> {target_path}'):
        shutil.copy(file_path, target_path)
        os.chmod(target_path,777)
    
    return target_path

def get_msa_templates_pipeline(cfg: DictConfig) -> Dict:
    use_precomputed_msas = True  # Assuming this is a constant or should be set globally
    

    if cfg.ramdisk.uniprot:
        cfg.db.uniprot=load_to_dev_shm(cfg.db.uniprot,keep=cfg.ramdisk.keep)
    
    if cfg.ramdisk.uniref90:
        cfg.db.uniref90=load_to_dev_shm(cfg.db.uniref90,keep=cfg.ramdisk.keep)

    if cfg.ramdisk.mgnify:
        cfg.db.mgnify=load_to_dev_shm(cfg.db.mgnify,keep=cfg.ramdisk.keep)

    template_searcher = hmmsearch.Hmmsearch(
        binary_path=resolve_bin_path(cfg.bin.hmmsearch, 'hmmsearch'),
        hmmbuild_binary_path=resolve_bin_path(cfg.bin.hmmbuild, 'hmmbuild'),
        database_path=cfg.db.pdb_seqres)

    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=cfg.template.mmcif_dir,
        max_template_date=cfg.template.max_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=resolve_bin_path(cfg.bin.kalign, 'kalign'),
        release_dates_path=None,
        obsolete_pdbs_path=cfg.template.obsolete_pdbs)

    monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=resolve_bin_path(cfg.bin.jackhmmer, 'jackhmmer'),
        hhblits_binary_path=resolve_bin_path(cfg.bin.hhblits, 'hhblits'),
        hhsearch_binary_path=resolve_bin_path(cfg.bin.hhsearch, 'hhsearch'),
        uniref90_database_path=cfg.db.uniref90,
        mgnify_database_path=cfg.db.mgnify,
        bfd_database_path=cfg.db.bfd,
        uniclust30_database_path=cfg.db.uniclust30,
        small_bfd_database_path=cfg.db.small_bfd,
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=cfg.use_small_bfd,
        use_precomputed_msas=use_precomputed_msas,
        nprocs=cfg.nproc_msa,
        mem=cfg.mem
        )



    prot_data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=resolve_bin_path(cfg.bin.jackhmmer, 'jackhmmer'),
        uniprot_database_path=cfg.db.uniprot,
        use_precomputed_msas=use_precomputed_msas)

    rna_monomer_data_pipeline = pipeline_rna.RNADataPipeline(
      hmmer_binary_path=resolve_bin_path(cfg.bin.nhmmer, 'nhmmer'),
      rfam_database_path=cfg.db.rfam,
      rnacentral_database_path=None,
      nt_database_path=None,     
      species_identifer_map_path=None,
      use_precomputed_msas=use_precomputed_msas)  

    rna_data_pipeline = pipeline_rna_multimer.RNADataPipeline(
      monomer_data_pipeline=rna_monomer_data_pipeline)

    return {
        'protein': prot_data_pipeline,
        'rna': rna_data_pipeline
    }
def ranking_all_predictions(output_dirs):
    ranking_score_path_map = {}
    for outpath in output_dirs:
        _results = preprocess.read_json(os.path.join(outpath, 'all_results.json'))
        _rank_score = _results['ranking_confidence']
        ranking_score_path_map[outpath] = _rank_score

    ranked_map = dict(sorted(ranking_score_path_map.items(), key=lambda x: x[1], reverse=True))
    rank_id = 1
    for outpath, rank_score in ranked_map.items():
        logging.debug("[ranking_all_predictions] Ranking score of %s: %.5f", outpath, rank_score)
        basename_prefix = os.path.basename(outpath).split('-pred-')[0]
        target_path = os.path.join(os.path.dirname(outpath), f'{basename_prefix}-rank{rank_id}')
        if os.path.exists(target_path) and os.path.isdir(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(outpath, target_path)
        rank_id += 1

@paddle.no_grad()
def evaluate(args, model: RunModel, batch):
    """evaluate a given dataset"""
    model.eval()       
        
    # inference
    def _forward_with_precision(batch):
        if args.precision == "bf16" or args.bf16_infer:
            black_list, white_list = get_custom_amp_list()
            with paddle.amp.auto_cast(enable=True,
                                        custom_white_list=white_list, 
                                        custom_black_list=black_list, 
                                        level=args.amp_level, 
                                        dtype='bfloat16'):
                return model(batch, compute_loss=False)
        elif args.precision == "fp32":
            return model(batch, compute_loss=False)
        else:
            raise ValueError("Please choose precision from bf16 and fp32! ")
        
    res = _forward_with_precision(batch)
    logging.info(f"Inference Succeeds...\n")
    return res


def postprocess_fn(entry_name, batch, results, output_dir, maxit_binary=None):
    """
        postprocess function for HF3 output.
            - batch. input data
            - results. model output
            - output_dir. to save output
            - maxit_binary. path to maxit binary
    """
    diff_results = results['diffusion_module']
    confidence_results = results['confidence_head']

    required_keys = copy.deepcopy(all_atom_pdb_save.required_keys_for_saving)
    required_keys += ['token_bonds_type', 'ref_element', 'is_ligand']
    required_keys = required_keys + ['atom_plddts']

    # 1 feat extraction
    common_feat = {k: batch['feat'][k][0]
            for k in required_keys if k in batch['feat']}
    common_feat.update(
        {k: batch['label'][k][0]
            for k in required_keys if k in batch['label']}
    )
    common_feat.update(
        {'atom_plddts': confidence_results['atom_plddts'][0]})

    ## NOTE: remove "UNK-"
    common_feat['all_ccd_ids'] = re.sub(r'UNK-\w*', 'UNK', common_feat['all_ccd_ids']).split()
    common_feat['all_atom_ids'] = str(common_feat['all_atom_ids']).split()

    ## asym_id start with 1
    common_feat['asym_id'] -= 1
    ## resid start with 1
    common_feat['residue_index'] += 1

    pred_dict = {
        "pos": diff_results['final_atom_positions'].numpy(),
        "mask": diff_results['final_atom_mask'].numpy(),
    }
    exp_dict = {
        "mask": batch['label']['all_atom_pos_mask'].numpy(),
    }

    atom_mask = np.logical_and(pred_dict["mask"] > 0, 
                exp_dict["mask"] > 0)[0]  # [N_atom]
    token_mask = batch['label']['all_centra_token_indice_mask'][0].numpy().astype('bool')
    # tensor to numpy
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            common_feat[feat_key] = common_feat[feat_key].numpy()
        if feat_key in ['residue_index', 'asym_id']:
            common_feat[feat_key] = common_feat[feat_key].astype(np.int32)

    def apply_mask(key, val):
        """ apply mask to val """
        val = np.array(val)
        if key in atom_level_keys or key in ['atom_plddts']:
            if key in ['ref_token2atom_idx']:
                return map_to_continuous_indices(val[atom_mask])
            return val[atom_mask]
        else:
            if key in ['token_bonds_type']:
                return val[token_mask, :][:, token_mask] 
            return val[token_mask]
    common_feat_masked = {k: apply_mask(k, v) for k, v in common_feat.items()}

    ## save prediction masked 
    pred_cif_path = f'{output_dir}/predicted_structure.cif'
    all_atom_pdb_save.prediction_to_mmcif(
        pred_dict["pos"][0][atom_mask], 
        common_feat_masked, 
        maxit_binary=maxit_binary, 
        mmcif_path=pred_cif_path)
    
    assert os.path.exists(pred_cif_path),\
              (f"pred: {pred_cif_path} not exists! please check it")


    #### NOTE: append some contexts to cif file, Now only support ligand-intra bond type.
    ## 1. license
    extra_infos = {'entry_id': entry_name, "global_plddt": float(confidence_results['mean_plddt'])}
    mmcif_writer.mmcif_meta_append(pred_cif_path, extra_infos)
    
    ## 2. post add ligand bond type;
    ## N_token, for ligand, N_token == N_atom
    ref_token2atom_idx = common_feat_masked['ref_token2atom_idx']
    is_ligand = common_feat_masked['is_ligand'].astype(bool) # N_token
    perm_is_ligand = is_ligand[ref_token2atom_idx].astype(bool)
    
    ccd_ids = common_feat_masked['all_ccd_ids'] # N_atom
    atom_ids = common_feat_masked['all_atom_ids'] # N_atom
    token_bond_type = common_feat_masked['token_bonds_type'] # N_token 
    bond_mat = token_bond_type[ref_token2atom_idx][:, ref_token2atom_idx] # N_token -> N_atom
    ligand_bond_type = bond_mat[perm_is_ligand][:, perm_is_ligand]
    index1, index2 = np.nonzero(ligand_bond_type)
    bonds = [(int(i), int(j), ligand_bond_type[i][j]) for i, j in zip(index1, index2) if i < j]
    ligand_atom_ids = atom_ids[perm_is_ligand]
    ligand_ccd_ids = ccd_ids[perm_is_ligand]

    contexts = {'_chem_comp_bond.comp_id': [], 
                '_chem_comp_bond.atom_id_1': [], 
                '_chem_comp_bond.atom_id_2 ': [],
                '_chem_comp_bond.value_order': []}
    for idx, (i, j, bd_type) in enumerate(bonds):
        _bond_type = INVERSE_ALLOWED_LIGAND_BONDS_TYPE_MAP[bd_type]
        contexts['_chem_comp_bond.comp_id'].append(ligand_ccd_ids[i])
        contexts['_chem_comp_bond.atom_id_1'].append(ligand_atom_ids[i])
        contexts['_chem_comp_bond.atom_id_2 '].append(ligand_atom_ids[j])
        contexts['_chem_comp_bond.value_order'].append(_bond_type)
        # contexts['_chem_comp_bond.pdbx_ordinal'].append(idx + 1)
    mmcif_writer.mmcif_append(pred_cif_path, contexts, rm_duplicates=True)
    #### NOTE: append some contexts to cif file


def get_display_results(batch, results):
    confidence_score_float_names = ['ptm', 'iptm', 'has_clash', 'mean_plddt', 'ranking_confidence']
    confidence_score_names = ['atom_plddts', 'pae']
    ## atom_plddts: [N_atom], pae: [N_token, N_token]
    required_atom_level_keys = atom_level_keys + ['atom_plddts']
    display_required_keys = ['all_ccd_ids', 'all_atom_ids', 
                            'ref_token2atom_idx', 'restype', 
                            'residue_index', 'asym_id',
                            'all_atom_pos_mask',]
    all_results = {k: [] for k in DISPLAY_RESULTS_KEYS}
    for k in confidence_score_float_names:
        all_results[k] = float(results['confidence_head'][k])

    diff_results = results['diffusion_module']
    # 1 feat extraction
    common_feat = {k: batch['feat'][k][0]
            for k in display_required_keys if k in batch['feat']}
    common_feat.update(
        {k: batch['label'][k][0]
            for k in  display_required_keys if k in batch['label']}
    )
    common_feat.update({k: results['confidence_head'][k][0]
                            for k in confidence_score_names})

    ## NOTE: remove "UNK-"
    common_feat['all_ccd_ids'] = re.sub(r'UNK-\w*', 'UNK', common_feat['all_ccd_ids']).split()
    common_feat['all_atom_ids'] = str(common_feat['all_atom_ids']).split()
    ## asym_id start with 1
    common_feat['asym_id'] -= 1
    ## resid start with 1
    common_feat['residue_index'] += 1

    pred_dict = {
        "pos": diff_results['final_atom_positions'].numpy(),
        "mask": diff_results['final_atom_mask'].numpy(),
    }
    exp_dict = {
        "mask": batch['label']['all_atom_pos_mask'].numpy(),
    }

    atom_mask = np.logical_and(pred_dict["mask"] > 0, exp_dict["mask"] > 0)[0]  # [N_atom] get valid atom
    token_mask = batch['label']['all_centra_token_indice_mask'][0].numpy().astype('bool') # get valid token
    # tensor to numpy
    for feat_key in common_feat:
        if isinstance(common_feat[feat_key], paddle.Tensor):
            common_feat[feat_key] = common_feat[feat_key].numpy()
        if feat_key in ['residue_index', 'asym_id']:
            common_feat[feat_key] = common_feat[feat_key].astype(np.int32)

    def apply_mask(key, val):
        """ apply mask to val """
        val = np.array(val)
        if key in required_atom_level_keys:
            if key in ['ref_token2atom_idx']:
                return map_to_continuous_indices(val[atom_mask])
            return val[atom_mask]
        else:
            if key in ['token_bonds_type', 'pae']:
                return val[token_mask, :][:, token_mask] 
            return val[token_mask]
    common_feat_masked = {k: apply_mask(k, v) for k, v in common_feat.items()}

    ## NOTE: save display results.
    ref_token2atom_idx = common_feat_masked['ref_token2atom_idx']
    chain_ids = common_feat_masked['asym_id'][ref_token2atom_idx] # N_token -> N_atom

    ## token-level
    all_results['pae'] = common_feat_masked['pae']
    for i in common_feat_masked['asym_id']:
        all_results['token_chain_ids'].append(all_atom_pdb_save.all_chain_ids[i])
    for i in common_feat_masked['residue_index']:
        all_results['token_res_ids'].append(i)

    ## atom-level
    all_results['atom_plddts'] = common_feat_masked['atom_plddts']
    all_results['atom_chain_ids'] = [all_atom_pdb_save.all_chain_ids[ca_i] for ca_i in chain_ids]

    return all_results


def save_result(entry_name, feature_dict, prediction, output_dir, maxit_bin):
    postprocess_fn(entry_name=entry_name,
                    batch=feature_dict, 
                    results=prediction,
                    output_dir=output_dir,
                    maxit_binary=maxit_bin)
    
    all_results = {k: [] for k in DISPLAY_RESULTS_KEYS}
    res = get_display_results(batch=feature_dict,results=prediction)
    
    for k in all_results:
        if k in res:
            all_results[k] = convert_to_json_compatible(res[k])

    with open(output_dir.joinpath('all_results.json'), 'w') as f:
        f.write(json.dumps(all_results, indent=4))
    
    root_path = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile(pathlib.Path(root_path).joinpath('LICENSE'), output_dir.joinpath('terms_of_use.md'))

def split_prediction(pred, rank):
    prediction = []
    feat_key_list = [pred[rk].keys() for rk in RETURN_KEYS]
    feat_key_table = dict(zip(RETURN_KEYS, feat_key_list))
    
    for i in range(rank):
        sub_pred = {}
        for rk in RETURN_KEYS:
            feat_keys = feat_key_table[rk]
            sub_feat = dict(zip(feat_keys, [pred[rk][fk][:, i] for fk in feat_keys]))
            sub_pred[rk] = sub_feat
    
        prediction.append(sub_pred)
    
    return prediction

@dataclass
class HelixFoldRunner:

    cfg: DictConfig

    model: RunModel =None
    model_config: DictConfig=None

    ccd_dict: Mapping=None
    msa_templ_data_pipeline_dict: Mapping=None

    def __post_init__(self) -> None:

        logging.set_verbosity(self.cfg.logging_level)
        ccd_preprocessed_path = self.cfg.db.ccd_preprocessed
        self.ccd_dict=load_ccd_dict(ccd_preprocessed_path)


        if self.cfg.msa_only == True:
            logging.warning(f'Model inference will be skipped because MSA-only mode is required.')
            logging.warning(f'Use CPU only')
            paddle.device.set_device("cpu")
            

        """main function"""
        new_einsum = os.getenv("FLAGS_new_einsum", True)
        print(f'>>> PaddlePaddle commit: {paddle.version.commit}')
        print(f'>>> FLAGS_new_einsum: {new_einsum}')
        print(f'>>> config:\n{self.cfg}')

        ## check maxit binary path
        maxit_binary=resolve_bin_path(self.cfg.other.maxit_binary,'maxit')
        
        RCSBROOT=os.path.join(os.path.dirname(maxit_binary), '..')
        os.environ['RCSBROOT']=RCSBROOT

        ## check obabel
        obabel_bin=resolve_bin_path(self.cfg.bin.obabel,'obabel')
        os.environ['OBABEL_BIN']=obabel_bin

        use_small_bfd = self.cfg.preset.preset == 'reduced_dbs'

        # fix to small bfd setting
        self.cfg.use_small_bfd=use_small_bfd

        if self.cfg.use_small_bfd:
            assert self.cfg.db.small_bfd is not None
        else:
            assert self.cfg.db.bfd is not None
            assert self.cfg.db.uniclust30 is not None

        ### Create model
        self.model_config = update_model_config(self.cfg.CONFIG_DIFFS)
        logging.warning(f'>>> Model config: \n{self.model_config}\n\n')

        self.model = RunModel(self.model_config)

        if (not self.cfg.weight_path is None) and (self.cfg.weight_path != ""):
            print(f"Load pretrain model from {self.cfg.weight_path}")
            pd_params = paddle.load(self.cfg.weight_path)
            
            has_opt = 'optimizer' in pd_params
            if has_opt:
                self.model.helixfold.set_state_dict(pd_params['model'])
            else:
                self.model.helixfold.set_state_dict(pd_params)

        logging.info('Getting MSA/Template Pipelines...')
        self.msa_templ_data_pipeline_dict = get_msa_templates_pipeline(cfg=self.cfg)

        
        if self.cfg.precision == "bf16" and self.cfg.amp_level == "O2":
            raise NotImplementedError("bf16 O2 is not supported yet.")
    
    def preprocess_json_entity(self, json_path, out_dir):
        all_entitys = preprocess.online_json_to_entity(json_path, out_dir, self.ccd_dict)
        if all_entitys is None:
            raise ValueError("The json file does not contain any valid entity.")
        else:
            logging.info("The json file contains %d valid entity.", len(all_entitys))
        
        return all_entitys
    def fold(self, entity: str):
        all_entitys = self.preprocess_json_entity(entity, self.cfg.output)
        
        ### Set seed for reproducibility
        seed = self.cfg.seed
        if seed is None:
            seed = np.random.randint(10000000)
        else:
            logging.warning('Seed is only used for reproduction')
        init_seed(seed)


        print(f"============ Data Loading ============")
        job_base = pathlib.Path(entity).stem
        output_dir_base = pathlib.Path(self.cfg.output).joinpath(job_base)

        expected_res=os.path.join(self.cfg.output, job_base, f'{job_base}-rank1','all_results.json')
        if os.path.isfile(expected_res):
            logging.warning(f'Skip {job_base} because {expected_res} exists')
            return


        msa_output_dir = output_dir_base.joinpath('msas')
        msa_output_dir.mkdir(parents=True, exist_ok=True)

        features_pkl = output_dir_base.joinpath('final_features.pkl')
        if features_pkl.exists() and not self.cfg.override:
            with open(features_pkl, 'rb') as f:
                logging.info(f'Load features from precomputed {features_pkl}')
                feature_dict = pickle.load(f)
        else:
            feature_dict = feature_processing_aa.process_input_json(
                            all_entitys, 
                            ccd_preprocessed_dict=self.ccd_dict,
                            msa_templ_data_pipeline_dict=self.msa_templ_data_pipeline_dict,
                            msa_output_dir=msa_output_dir)

            # save features
            with open(features_pkl, 'wb') as f:
                pickle.dump(feature_dict, f, protocol=4)

        if self.cfg.msa_only == True:
            logging.warning(f'Model inference is skipped because MSA-only mode is required.')
            return

        feature_dict['feat'] = batch_convert(feature_dict['feat'], add_batch=True)
        feature_dict['label'] = batch_convert(feature_dict['label'], add_batch=True)
        
        print(f"============ Start Inference ============")
        
        infer_times = self.cfg.infer_times
        if self.cfg.diff_batch_size > 0:
            self.model_config.model.heads.diffusion_module.test_diff_batch_size = self.cfg.diff_batch_size
        diff_batch_size = self.model_config.model.heads.diffusion_module.test_diff_batch_size 
        logging.info(f'Inference {infer_times} Times...')
        logging.info(f"Diffusion batch size {diff_batch_size}...\n")
        all_pred_path = []
        for infer_id in range(infer_times):
            
            with timing(f'{infer_id}-th inference'):
                prediction = evaluate(self.cfg, self.model, feature_dict)
                
                # save result
                prediction = split_prediction(prediction, diff_batch_size)
                for rank_id in range(diff_batch_size):
                    json_name = job_base + f'-pred-{str(infer_id + 1)}-{str(rank_id + 1)}'
                    output_dir = pathlib.Path(output_dir_base).joinpath(json_name)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    save_result(entry_name=job_base,
                                feature_dict=feature_dict,
                                prediction=prediction[rank_id],
                                output_dir=output_dir, 
                                maxit_bin=self.cfg.other.maxit_binary)
                    all_pred_path.append(output_dir)
        
        # final ranking
        print(f'============ Ranking ! ============')
        ranking_all_predictions(all_pred_path)
        print(f'============ Inference finished ! ============')
    

    def cleanup_ramdisk(self, ramdisk_path: str = "/dev/shm"):
        if self.cfg.ramdisk.keep:
            logging.info('Keep DB in RAM disk.')
            return
        logging.warning('Removing all DBs from RAM disk ....')
        for db_fasta in [db for db in (self.cfg.db.uniprot, self.cfg.db.uniref90, self.cfg.db.mgnify,) if db.startswith(ramdisk_path)]:
            try:
                os.unlink(db_fasta)
            except Exception as e:
                logging.error(f"Failed to delete {db_fasta} from ram disk. Reason: {e}")
            

@hydra.main(version_base=None, config_path=os.path.join(script_path,'config',),config_name='helixfold')
def main(cfg: DictConfig):
    
    hf_runner=HelixFoldRunner(cfg=cfg)

    if os.path.isfile(cfg.input):
        logging.info(f'Starting inference on {cfg.input}')
        try:
            hf_runner.fold(cfg.input)
        except (ValueError, AssertionError, RuntimeError, FileNotFoundError, MemoryError) as e :
            logging.error(f'Error processing {cfg.input}: {e}')
    elif os.path.isdir(cfg.input):
        logging.info(f'Starting inference on all files in {cfg.input}')
        for f in [i for i in os.listdir(cfg.input) if any(i.endswith(p) for p in ['json', 'jsonl', 'json.gz', 'jsonl.gz'])]:
            logging.info(f'Processing  {f}')
            try:
                hf_runner.fold(os.path.join(cfg.input,f))
            except (ValueError, AssertionError, RuntimeError, FileNotFoundError, MemoryError, OSError) as e :
                logging.error(f'Error processing {f}: {e}')
                continue


    return hf_runner.cleanup_ramdisk()
    


@hydra.main(version_base=None, config_path=os.path.join(script_path,'config',),config_name='helixfold')
def check_ligand(cfg: DictConfig):
    from helixfold.utils.preprocess import ligand_convert
    ## check obabel
    obabel_bin=resolve_bin_path(cfg.bin.obabel,'obabel')
    os.environ['OBABEL_BIN']=obabel_bin
    ccd_preprocessed_path = cfg.db.ccd_preprocessed
    ccd_dict=load_ccd_dict(ccd_preprocessed_path)


    sm_ligand_fp: Union[list[str], str]=cfg.ligand
    if isinstance(sm_ligand_fp, str):
        sm_ligand_fp=[sm_ligand_fp]

    for sm in sm_ligand_fp:

        if len(sm) <= 3: 
            ccd_id=sm
            if ccd_id in ccd_dict:
                logging.info(f'Atoms in {ccd_id}: {ccd_dict[ccd_id]}')
                continue
            else:
                raise KeyError(f'Failed to load CCD key `{sm}` from CCD dict.')
        

        ligand_type='smiles' if not os.path.isfile(sm) else os.path.basename(sm).split('.')[-1]

        logging.info(f'Guessed ligand input type: {ligand_type}')
        ligand_entity=ligand_convert(items={
            'type':'ligand',
            ligand_type: sm,
            'name': 'UNK',
            'count': 1
        })

        logging.info(f'Atoms in {sm} ({ligand_type}): {ligand_entity.extra_mol_infos}')


if __name__ == '__main__':
    main()
