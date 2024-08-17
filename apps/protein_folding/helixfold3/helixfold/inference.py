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
import re
import os
import copy
import random
import paddle
import json
import pickle
import pathlib
import shutil
import logging
import numpy as np
import shutil

from omegaconf import DictConfig
import hydra

from helixfold.common import all_atom_pdb_save
from helixfold.model import config, utils
from helixfold.data import pipeline_parallel as pipeline
from helixfold.data import pipeline_multimer_parallel as pipeline_multimer
from helixfold.data import pipeline_rna_parallel as pipeline_rna
from helixfold.data import pipeline_rna_multimer
from helixfold.data.utils import atom_level_keys, map_to_continuous_indices
from helixfold.utils.model import RunModel
from helixfold.data.tools import hmmsearch
from helixfold.data import templates
from helixfold.utils.utils import get_custom_amp_list
from helixfold.utils.misc import set_logging_level
from typing import Dict
from helixfold.infer_scripts import feature_processing_aa, preprocess
from helixfold.infer_scripts.tools import mmcif_writer


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

logger = logging.getLogger(__file__)

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

def preprocess_json_entity(json_path, out_dir):
    all_entitys = preprocess.online_json_to_entity(json_path, out_dir)
    if all_entitys is None:
        raise ValueError("The json file does not contain any valid entity.")
    else:
        logger.info("The json file contains %d valid entity.", len(all_entitys))
    
    return all_entitys

def convert_to_json_compatible(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_compatible(i) for i in obj]
    else:
        return obj

def resolve_bin_path(cfg_path: str, default_binary_name: str)-> str:
    """Helper function to resolve the binary path."""
    if cfg_path and os.path.isfile(cfg_path):
        return cfg_path

    if cfg_val:=shutil.which(default_binary_name):
        logging.warning(f'Using resolved {default_binary_name}: {cfg_val}')
        return cfg_val

    raise FileNotFoundError(f"Could not find a proper binary path for {default_binary_name}: {cfg_path}.")

def get_msa_templates_pipeline(cfg: DictConfig) -> Dict:
    use_precomputed_msas = True  # Assuming this is a constant or should be set globally

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
        use_precomputed_msas=use_precomputed_msas)

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
        logger.debug("[ranking_all_predictions] Ranking score of %s: %.5f", outpath, rank_score)
        basename_prefix = os.path.basename(outpath).split('-pred-')[0]
        target_path = os.path.join(os.path.dirname(outpath), f'{basename_prefix}-rank{rank_id}')
        if os.path.exists(target_path) and os.path.isdir(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(outpath, target_path)
        rank_id += 1

@paddle.no_grad()
def eval(args, model, batch):
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
    logger.info(f"Inference Succeeds...\n")
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


@hydra.main(version_base=None, config_path=os.path.join(script_path,'config',),config_name='helixfold')
def main(cfg: DictConfig):
    set_logging_level(cfg.logging_level)

    if cfg.msa_only == True:
        logging.warning(f'Model inference will be skipped because MSA-only mode is required.')
        logging.warning(f'Use CPU only')
        paddle.device.set_device("cpu")
        

    """main function"""
    new_einsum = os.getenv("FLAGS_new_einsum", True)
    print(f'>>> PaddlePaddle commit: {paddle.version.commit}')
    print(f'>>> FLAGS_new_einsum: {new_einsum}')
    print(f'>>> config:\n{cfg}')

    all_entitys = preprocess_json_entity(cfg.input, cfg.output)
    ## check maxit binary path
    maxit_binary=resolve_bin_path(cfg.other.maxit_binary,'maxit')
    
    RCSBROOT=os.path.join(os.path.dirname(maxit_binary), '..')
    os.environ['RCSBROOT']=RCSBROOT

    ## check obabel
    obabel_bin=resolve_bin_path(cfg.bin.obabel,'obabel')
    os.environ['OBABEL_BIN']=obabel_bin

    ### Set seed for reproducibility
    seed = cfg.seed
    if seed is None:
        seed = np.random.randint(10000000)
    else:
        logger.warning('Seed is only used for reproduction')
    init_seed(seed)

    use_small_bfd = cfg.preset.preset == 'reduced_dbs'
    setattr(cfg, 'use_small_bfd', use_small_bfd)
    if use_small_bfd:
        assert cfg.db.small_bfd is not None
    else:
        assert cfg.db.bfd is not None
        assert cfg.db.uniclust30 is not None

    logger.info('Getting MSA/Template Pipelines...')
    msa_templ_data_pipeline_dict = get_msa_templates_pipeline(cfg=cfg)
        
    ### Create model
    model_config = config.model_config(cfg.CONFIG_DIFFS)
    logging.warning(f'>>> Model config: \n{model_config}\n\n')

    model = RunModel(model_config)

    if (not cfg.weight_path is None) and (cfg.weight_path != ""):
        print(f"Load pretrain model from {cfg.weight_path}")
        pd_params = paddle.load(cfg.weight_path)
        
        has_opt = 'optimizer' in pd_params
        if has_opt:
            model.helixfold.set_state_dict(pd_params['model'])
        else:
            model.helixfold.set_state_dict(pd_params)

    
    
    if cfg.precision == "bf16" and cfg.amp_level == "O2":
        raise NotImplementedError("bf16 O2 is not supported yet.")

    print(f"============ Data Loading ============")
    job_base = pathlib.Path(cfg.input).stem
    output_dir_base = pathlib.Path(cfg.output).joinpath(job_base)
    msa_output_dir = output_dir_base.joinpath('msas')
    msa_output_dir.mkdir(parents=True, exist_ok=True)

    features_pkl = output_dir_base.joinpath('final_features.pkl')
    if features_pkl.exists() and not cfg.override:
        with open(features_pkl, 'rb') as f:
            logging.info(f'Load features from precomputed {features_pkl}')
            feature_dict = pickle.load(f)
    else:
        feature_dict = feature_processing_aa.process_input_json(
                        all_entitys, 
                        ccd_preprocessed_path=cfg.db.ccd_preprocessed,
                        msa_templ_data_pipeline_dict=msa_templ_data_pipeline_dict,
                        msa_output_dir=msa_output_dir)

        # save features
        with open(features_pkl, 'wb') as f:
            pickle.dump(feature_dict, f, protocol=4)

    if cfg.msa_only == True:
        logging.warning(f'Model inference is skipped because MSA-only mode is required.')
        exit()

    feature_dict['feat'] = batch_convert(feature_dict['feat'], add_batch=True)
    feature_dict['label'] = batch_convert(feature_dict['label'], add_batch=True)
    
    print(f"============ Start Inference ============")
    
    infer_times = cfg.infer_times
    if cfg.diff_batch_size > 0:
        model_config.model.heads.diffusion_module.test_diff_batch_size = cfg.diff_batch_size
    diff_batch_size = model_config.model.heads.diffusion_module.test_diff_batch_size 
    logger.info(f'Inference {infer_times} Times...')
    logger.info(f"Diffusion batch size {diff_batch_size}...\n")
    all_pred_path = []
    for infer_id in range(infer_times):
        
        logger.info(f'Start {infer_id}-th inference...\n')
        prediction = eval(cfg, model, feature_dict)
        
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
                        maxit_bin=cfg.other.maxit_binary)
            all_pred_path.append(output_dir)
    
    # final ranking
    print(f'============ Ranking ! ============')
    ranking_all_predictions(all_pred_path)
    print(f'============ Inference finished ! ============')

if __name__ == '__main__':
    main()
