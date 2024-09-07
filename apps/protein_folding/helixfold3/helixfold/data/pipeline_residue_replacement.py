from dataclasses import dataclass
from absl import logging

from helixfold.common import residue_constants

@dataclass(frozen=True)
class ResidueReplacement:
    chain: str
    residue_index: int
    old_residue: str
    new_residue: str

    def __post_init__(self):
        if self.old_residue not in residue_constants.restype_3to1:
            raise ValueError(f'Invalid old residue: {self.old_residue}')
        if len(self.new_residue) >3:
            raise ValueError("New residue should be 1 - 3 letters")
        

    def __str__(self) -> str:
        return f'Chain {self.chain}: ({self.residue_index}) | {self.old_residue} -> {self.new_residue} '


def parse_residue_replacement(modres:str):
    replacements: list[ResidueReplacement]=[]
    for _modres in modres.split(';'):
        if not _modres:
            continue
        modrespart=_modres.split(',')

        if not len(modrespart) ==4:
            raise ValueError(f'Invalid replacement format: {_modres}. Expected 4 fields per replacenment.')
        
        chain, resi, old,new=modrespart
        replacements.append(ResidueReplacement(chain, int(resi), old,new))
    
    logging.info(f'Added {len(replacements)} Residue Replacements.\n{replacements}')
    return replacements

