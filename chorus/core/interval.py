from dataclasses import dataclass, field 
from typing import TypeVar

@dataclass
class GenomeInterval: 
    chrom: int 
    start: int 
    end: int 
    reference: str
    @property
    def length(self) -> int:
        return self.end - self.start 

@dataclass
class SequenceInterval:
    start: int
    end: int 
    sequence: str 
    
    @property
    def length(self) -> int:
        return self.end - self.start 
    
    @property
    def chrom(self) -> str:
        return None

class ReplacementInterval(SequenceInterval):
    pass 

class InsertionInterval(SequenceInterval):
    pass 

@dataclass 
class NsInterval: # padding
    length: int

    @property
    def chrom(self) -> str:
        return None

@dataclass 
class DeletionInterval: # padding
    length: int

    @property
    def chrom(self) -> str:
        return None

SingleInterval = TypeVar('SingleInterval', SequenceInterval, GenomeInterval, NsInterval)

@dataclass 
class JointInterval:
    intervals: tuple[SingleInterval]
    start: int = field(init=False)
    end: int = field(init=False)

    @property
    def length(self) -> int:
        return sum(i.length for i in self.intervals)

    @property
    def chrom(self) -> str:
        chroms = []
        for i in self.intervals:
            if i.chrom is not None:
                chroms.append(i.chrom)
        chroms = list(set(chroms))
        if len(chroms) != 1:
            raise NotImplementedError("For now chorus can't store predictions for different chromosomes in the same oracleprediction object")
        return chroms[0]

    def __post_init__(self):
        starts = []
        ends = [ ]
        for interv in self.intervals:
            if isinstance(interv, GenomeInterval):
                starts.append(interv.start)
                ends.append(interv.end)
        if len(starts) == 0: # no genome_intervals
            for interv in self.intervals:
                if isinstance(interv, JointInterval):
                    starts.append(interv.start)
                    ends.append(interv.end)
        if len(starts) == 0:
            self.start = 0
            self.end = 0
        else:
            self.start = min(starts)
            self.end = max(starts)

Interval = TypeVar('Interval', SequenceInterval, GenomeInterval, JointInterval)