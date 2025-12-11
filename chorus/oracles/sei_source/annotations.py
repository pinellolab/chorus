# This submodule is safe for importing in base environment

import logging 

from typing import ClassVar
from dataclasses import dataclass
from .utils import unique, list_match, match
from typing import Tuple
from .exceptions import SeiError

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SeiClass:
    label: str
    name: str
    group: str
    rank: int

    ID_PREF: str = "CA"

    def match(self,
              pat: Tuple[str | None, str | None],
              exact: bool = True, 
              regex: bool = True,
              case: bool = True) -> bool:
        pat_name, pat_group = pat
        if pat_name is not None:
            name_match = match(pat=pat_name,
                            text=self.name,
                            exact=exact,
                            regex=regex,
                            case=case)
        else:
            name_match = True
        
        if pat_group is not None:
            group_match = match(pat=pat_group,
                            text=self.group,
                            exact=exact,
                            regex=regex,
                            case=case)
        else:
            group_match = True
        return name_match and group_match
    
    def __str__(self) -> str:
        return f"{self.ID_PREF}#{self.label}@{self.name}@{self.group}@{self.rank}"
    
    @classmethod
    def from_str(cls, str_cl: str) -> 'SeiClass':
        pref, str_cl = str_cl.split("#", maxsplit=1)
        if pref != cls.ID_PREF:
            raise SeiError(f"Invalid class ID: {str_cl}")
        label, name, group, rank = str_cl.split("@")
        rank = int(rank)
        return cls(label=label, name=name, group=group, rank=rank)
    
    @classmethod
    def is_id(cls, str_cl: str) -> bool:
        pref, rest = str_cl.split("#", maxsplit=1)
        return pref == cls.ID_PREF and rest.count("@") == 3

@dataclass
class SeiClassesList:
    classes: dict[SeiClass, int]

    @classmethod
    def load(cls, path: str) -> 'SeiClassesList':
        classes = {}
        with open(path, 'r') as inp:
            inp.readline() # skip header
            for ind, line in enumerate(inp):
                label, name, rank, group = line.strip().split("\t")
                rank = int(rank)
                ca = SeiClass(label=label, name=name, rank=rank, group=group)
                classes[ca] = ind

        return cls(classes)
    
    def list_class_types(self, rem_duplicates: bool = True) -> list[str]:
        names = [ca.name for ca in self.classes.keys()]
        if rem_duplicates:
            names = unique(names)
        return names

    def list_group_types(self, rem_duplicates: bool = True) -> list[str]:
        groups = [ca.group for ca in self.classes.keys()]
        if rem_duplicates:
            groups = unique(groups)
        return groups
  
    
    def retrieve_class_types(self, 
                             pat: str, 
                             exact: bool=False, 
                             regex: bool=True, 
                             case: bool=False,
                             return_names: bool = False)-> list[int | tuple[int, str]]:
        '''
        get indices and (optionally) names of class types matching pattern 
        ''' 
        names = self.list_class_types(rem_duplicates=False)
        matched = list_match(pat,
                             texts=names, 
                             exact=exact, 
                             regex=regex,
                             case=case, 
                             only_indices=not return_names)
        return matched
    
    def retrieve_group_types(self, 
                             pat: str, 
                             exact: bool=False, 
                             regex: bool=True, 
                             case: bool=False,
                             return_names: bool = False) -> list[int | tuple[int, str]]:
        '''
        get indices and (optionally) names of group types matching pattern 
        ''' 
        groups = self.list_group_types(rem_duplicates=False)
        matched = list_match(pat,
                             texts=groups, 
                             exact=exact, 
                             regex=regex,
                             case=case, 
                             only_indices=not return_names)
        return matched
    
    def select_classes(self,
                      pats: list[Tuple[str | None, str | None]] | str,
                      exact: bool=False, 
                      regex: bool=True, 
                      case: bool=False) -> list[SeiClass]:
        
        extended_pats = []
        for p in pats:
            if isinstance(p, str):
                extended_pats.append((p, None))
                extended_pats.append((None, p))
            else:
                extended_pats.append(p)
        
        matches = []
        for cl in self.classes.keys():
            for p in extended_pats:
                if cl.match(p, exact=exact, regex=regex, case=case): 
                    matches.append(cl)
                    break
        return matches
    
    def cl2ind(self, cls_lst: list[SeiClass]) -> list[int]:
        return [self.classes[cl] for cl in cls_lst]
    

@dataclass(frozen=True)
class SeiTarget:
    celltype: str
    assay: str
    sei_id: str 
    rep_ind: int = 0

    ID_PREF: str = 'TA'

    def match(self,
              pat: Tuple[str | None, str | None],
              exact: bool = True, 
              regex: bool = True,
              case: bool = True) -> bool:
        pat_assay, pat_celltype = pat
        if pat_assay is not None:
            assay_match = match(pat=pat_assay,
                            text=self.assay,
                            exact=exact,
                            regex=regex,
                            case=case)
        else:
            assay_match = True
        
        if pat_celltype is not None:
            celltype_match = match(pat=pat_celltype,
                                   text=self.celltype,
                                   exact=exact,
                                   regex=regex,
                                   case=case)
        else:
            celltype_match = True
        return assay_match and celltype_match
    
    def __str__(self) -> str:
        if self.rep_ind == 0:
            return f"{self.ID_PREF}#{self.celltype}@{self.assay}@{self.sei_id}"
        else:
            return f"{self.ID_PREF}#{self.celltype}@{self.assay}@{self.sei_id}@{self.rep_ind}"
    
    @classmethod
    def from_str(cls, str_ta: str) -> 'SeiTarget':
        pref, str_ta = str_ta.split("#", maxsplit=1)
        if pref != cls.ID_PREF:
            raise SeiError(f"Invalid target ID: {str_ta}")
        celltype, assay, sei_id, *rep_ind = str_ta.split("@")
        if len(rep_ind) == 0:
            rep_ind = 0
        else:
            rep_ind = int(rep_ind[0])

        return cls(celltype=celltype, assay=assay, sei_id=sei_id, rep_ind=rep_ind)

    @classmethod
    def is_id(cls, str_cl: str) -> bool:
        pref, rest = str_cl.split("#", maxsplit=1)
        return pref == cls.ID_PREF and (rest.count("@") == 2 or rest.count("@") == 3)

@dataclass
class SeiTargetList:
    targets: dict[SeiTarget, int]

    @classmethod
    def load(cls, path: str) -> 'SeiTargetList':
        targets = {}
        with open(path, 'r') as inp:
            for ind, line in enumerate(inp):
                line = line.strip()
                celltype, assay, sei_id, *rep_ind = line.split(" | ")
                if len(rep_ind) == 1:
                    rep_ind = int(rep_ind[0])
                elif len(rep_ind) > 1:
                    raise ValueError(f"Wrong format: {line}")
                else:
                    rep_ind = 0 
                ta = SeiTarget(celltype=celltype, assay=assay, sei_id=sei_id, rep_ind=rep_ind)
                targets[ta] = ind
                
        return cls(targets)

    def list_assay_types(self, rem_duplicates: bool=True) -> list[str]:
        as_tp = [ta.assay for ta in self.targets.keys()]
        if rem_duplicates:
            as_tp = unique(as_tp)
        return as_tp

    def list_cell_types(self, rem_duplicates: bool=True) -> list[str]:
        cl_tp = [ta.celltype for ta in self.targets.keys()]
        if rem_duplicates:
            cl_tp = unique(cl_tp)
        return cl_tp
    
    def retrieve_assay_types(self, 
                             pat: str, 
                             exact: bool=False, 
                             regex: bool=True, 
                             case: bool=False,
                             return_names: bool = False) -> list[int | tuple[int, str]]:
        '''
        get indices and (optionally) names of assay types matching pattern 
        ''' 
        as_tp = self.list_assay_types(rem_duplicates=False)
        matched = list_match(pat,
                             texts=as_tp, 
                             exact=exact, 
                             regex=regex,
                             case=case, 
                             only_indices=not return_names)
        return matched
    
    def retrieve_cell_types(self, 
                             pat: str, 
                             exact: bool=False, 
                             regex: bool=True, 
                             case: bool=False,
                             return_names: bool = False) -> list[int | tuple[int, str]]:
        '''
        get indices and (optionally) names of cell types matching pattern 
        ''' 
        as_cl = self.list_cell_types(rem_duplicates=False)
        matched = list_match(pat,
                             texts=as_cl, 
                             exact=exact, 
                             regex=regex,
                             case=case, 
                             only_indices=not return_names)
        return matched
    

    def select_targets(self,
                       pats: list[Tuple[str | None, str | None]] | str,
                       exact: bool=False, 
                       regex: bool=True, 
                       case: bool=False) -> list[SeiClass]:
        extended_pats = []
        for p in pats:
            if isinstance(p, str):
                extended_pats.append((p, None))
                extended_pats.append((None, p))
            else:
                extended_pats.append(p)

        matches = []
        for cl in self.targets.keys():
            for p in extended_pats:
                if cl.match(p, exact=exact, regex=regex, case=case): 
                    matches.append(cl)
                    break
        return matches

    def targets2inds(self, tgt_list: list[SeiTarget]) -> list[int]:
        return [self.targets[ta] for ta in tgt_list]