import re 
import numpy as np

from typing import Hashable, List, Sequence
    
def unique(l: Sequence[Hashable]) -> list:
    return list(dict.fromkeys(l))


import re 

def match(pat: str, text: str, exact: bool=False, regex: bool=False, case: bool=False):
    if not regex:
        if not case:
            text = text.lower()
            pat = pat.lower()
        if exact:
            return pat == text
        else:
            return pat in text 
    else:
        if not case:
            flags = re.IGNORECASE
        else:
            flags = 0
        if exact:
            return re.fullmatch(pat, text, flags=flags) is not None
        else:
            return re.search(pat, text, flags=flags) is not None


def list_match(pat: str, 
               texts: list[str], 
               exact: bool=False, 
               regex: bool=True, 
               case: bool=False,
               only_indices: bool = False) -> list[int | tuple[int, str]]:
    matched = []
    for ind, t in enumerate(texts):
        if match(pat, t, exact=exact, regex=regex, case=case):
            if only_indices:
                matched.append(ind)
            else:
                matched.append( (ind, t))
    return matched

def gather_with_nones(source: np.ndarray, inds: List[int | None], default=np.nan) -> np.ndarray:
        selected = np.zeros( (len(inds),) , dtype=source.dtype)
        for i, ind in enumerate(inds):
            if ind is None:
                selected[i] = default
            else:
                selected[i] = source[i]
        return selected