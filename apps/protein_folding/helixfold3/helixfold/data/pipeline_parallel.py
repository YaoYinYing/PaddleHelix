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

"""Functions for building the input features for the HelixFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union
from absl import logging
from helixfold.common import residue_constants
from helixfold.data import msa_identifiers
from helixfold.data import parsers
from helixfold.data import templates
from helixfold.data.tools import hhblits
from helixfold.data.tools import hhsearch
from helixfold.data.tools import hmmsearch
from helixfold.data.tools import jackhmmer
import numpy as np
from joblib import Parallel, delayed
# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]

class MsaRunner(Protocol):
    n_cpu: int
    
    def query(self, input_fasta_path: str) -> Sequence[Mapping[str, Any]]:
        """Runs the MSA tool on the input fasta file."""
        ...

def check_used_ncpus(used: list[int]):
   ncpus_sum=sum(used)
   if ncpus_sum >os.cpu_count():
      logging.warning(f"The number of used CPUs({ncpus_sum}) is larger than the number of available CPUs({os.cpu_count()}).")

def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features

# Yinying edited here to change various position args as one tuple for multiprocess tests
def run_msa_tool(args: Tuple[MsaRunner, str, str, str, bool, int]) -> Mapping[str, Any]:
    if args == None:
        return None
    if (len_args:=len(args))!=6:
       raise ValueError(f'MsaRunner must have exactly 6 arguments but got {len_args}')

    (msa_runner, input_fasta_path, msa_out_path,
      msa_format, use_precomputed_msas,
      max_sto_sequences) = args
    #print(args)
    """Runs an MSA tool, checking if output already exists first."""
    if not use_precomputed_msas or not os.path.exists(msa_out_path):
        if msa_format == 'sto' and max_sto_sequences > 0:
            result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
        else:
            result = msa_runner.query(input_fasta_path)[0]
        with open(msa_out_path, 'w') as f:
            f.write(result[msa_format])
    else:
        result=read_msa_result(msa_out_path,msa_format,max_sto_sequences)
    return result

def read_msa_result(msa_out_path,msa_format,max_sto_sequences):
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences > 0:
        precomputed_msa = parsers.truncate_stockholm_msa(
            msa_out_path, max_sto_sequences)
        result = {'sto': precomputed_msa}
    else:
        with open(msa_out_path, 'r') as f:
            result = {msa_format: f.read()}
    return result




class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               use_precomputed_msas: bool = False,
               nprocs: Mapping[str, int] = {
                  'hhblits': 16,
                  'jackhmmer': 8,
               }):
    """Initializes the data pipeline. Constructs a feature dict for a given FASTA file."""
    self._use_small_bfd = use_small_bfd
    self.nprocs=nprocs
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path, n_cpu=self.nprocs.get('jackhmmer', 8))
    if use_small_bfd:
      self.bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path, n_cpu=self.nprocs.get('jackhmmer', 8))
    else:
      self.bfd_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path], n_cpu=self.nprocs.get('hhblits', 8))
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path, n_cpu=self.nprocs.get('jackhmmer', 8))
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas

  def parallel_msa_joblib(self, func, input_args: list):
    return Parallel(len(input_args),verbose=100)(delayed(func)(args) for args in input_args)


  def process(self, input_fasta_path: str, msa_output_dir: str,other_args: Optional[tuple] = None) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)


    msa_tasks: list[Tuple[MsaRunner, str, str, str, bool, int]] = []
    """uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    jackhmmer_uniref90_result = run_msa_tool(
        msa_runner=self.jackhmmer_uniref90_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=uniref90_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.uniref_max_hits)"""
    msa_tasks.append((
          self.jackhmmer_uniref90_runner,
          input_fasta_path,
          os.path.join(msa_output_dir, 'uniref90_hits.sto'),
          'sto',
          self.use_precomputed_msas,
          self.uniref_max_hits))
    """mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    jackhmmer_mgnify_result = run_msa_tool(
        msa_runner=self.jackhmmer_mgnify_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=mgnify_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.mgnify_max_hits)"""
    msa_tasks.append((self.jackhmmer_mgnify_runner,                                                                                                     
           input_fasta_path,                                                                                                                    
           os.path.join(msa_output_dir, 'mgnify_hits.sto'),                                                                                  
           'sto',                                                                                                                               
           self.use_precomputed_msas,
           self.mgnify_max_hits))


    msa_tasks.append((
        self.bfd_runner,
        input_fasta_path,
        os.path.join(msa_output_dir, 'small_bfd_hits.sto' if self._use_small_bfd else 'bfd_uniclust_hits.a3m'),
        'sto' if self._use_small_bfd else 'a3m',
        self.use_precomputed_msas,
        0))
    
    msa_tasks.append(other_args)

    check_used_ncpus(used=[mask[0].n_cpu for mask in msa_tasks if hasattr(mask[0], 'n_cpu')])

    [
        jackhmmer_uniref90_result,
        jackhmmer_mgnify_result,
        bfd_result,
        
    ] = self.parallel_msa_joblib(func=run_msa_tool,
                          input_args=msa_tasks)

    msa_for_templates = jackhmmer_uniref90_result['sto']
    msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(msa_for_templates)

    if self.template_searcher.input_format == 'sto':
      pdb_templates_result = self.template_searcher.query(msa_for_templates)
    elif self.template_searcher.input_format == 'a3m':
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
      pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
    else:
      raise ValueError('Unrecognized template input format: '
                       f'{self.template_searcher.input_format}')

    pdb_hits_out_path = os.path.join(
        msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
    with open(pdb_hits_out_path, 'w') as f:
      f.write(pdb_templates_result)

    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
    mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])

    pdb_template_hits = self.template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=input_sequence)

    if self._use_small_bfd:
        bfd_msa = parsers.parse_stockholm(bfd_result['sto'])
    else:
        bfd_msa = parsers.parse_a3m(bfd_result['a3m'])

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        hits=pdb_template_hits,
        query_pdb_code=None,
        query_release_date=None)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))

    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    return {**sequence_features, **msa_features, **templates_result.features}
