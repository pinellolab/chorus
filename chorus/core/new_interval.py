import numpy as np 
import pysam

from dataclasses import dataclass, field
from dataclasses import replace as dt_replace
from typing import ClassVar, TypeVar
from Bio.Align import Alignment
from Bio import Align
from itertools import groupby
from copy import copy


def ilength(c):
    cnt = 0
    for _ in c:
        cnt += 1
    return cnt 

class IntervalException(Exception):
    pass

@dataclass(frozen=True)
class GenomeRef:
    chrom: str
    start: int
    end: int  # 0-based, end-exclusive
    fasta: str = field(repr=False) # path to fasta file
    extendible: bool = field(default=True, repr=False) # whether we can extend genome region using surrounding context
    hash_sequence: bool = field(default=True, repr=False) # # set to False to decrease memory usage

    def __len__(self) -> int:
        return self.end - self.start

    def __post_init__(self):
        object.__setattr__(self, '_sequence', None)

    @property
    def sequence(self) -> str:
        if self.start == self.end:
            return ''
        if self._sequence is None:
            from ..utils.sequence import extract_sequence
            if self.hash_sequence:                
                sequence = extract_sequence((self.chrom, self.start, self.end), self.fasta)
                object.__setattr__(self, '_sequence', sequence)
            else:
                return extract_sequence((self.chrom, self.start, self.end), self.fasta)
        return self._sequence

    def slop(self, extension_needed: int, how: str = 'both') ->  'GenomeRef':
        with pysam.FastaFile(self.fasta) as fasta:
            # Get chromosome length
            chrom_length = fasta.get_reference_length(self.chrom)
        if how == 'left':
            extend_left = extension_needed
        elif how == 'right':
            extend_left = 0
        elif how == 'both':
            extend_left = extension_needed // 2
        extend_right = extension_needed - extend_left    
        extended_start = max(0, self.start - extend_left)
        extended_end = min(chrom_length, self.end + extend_right)

        return dt_replace(self,
                          start=extended_start,
                          end=extended_end)
    
    def extend(self, length: int, how: str) -> 'GenomeRef':
        if not self.extendible:
            raise IntervalException("This genome reference entry can't be extended")
        extension_needed = length - len(self)
        return self.slop(extension_needed=extension_needed,
                         how=how)

    def zero(self) -> 'GenomeRef':
        return dt_replace(self,
                         start=0, 
                         end=0)

    def trim(self, left: int, right: int) -> 'GenomeRef':
        return dt_replace(self,
                          start=min(self.start+left, self.end),
                          end=max(self.end-right, 0))

    def slice(self, start: int, end: int) -> 'Sequence':
        return dt_replace(self, start=self.start+start, end=min(self.start + end, self.end) )
    
    def __getitem__(self, item) -> 'str | Sequence':
        if isinstance(item, int):
            return self.sequence[item]
        elif isinstance(item, slice):
            if item.step is not None and  item.step != 1:
                raise IntervalException("Sequence object doesn't support slicing with step different from 1")
            if item.start is None:
                start = 0
            else:
                start = item.start
            end = item.stop
            if start < 0 or end < 0:
                raise IntervalException("Interval object doesn't support slicing with negative indices")
            return self.slice(start, end)
        else:
            raise IntervalException(f"Unsupported indexing item: {item}")


@dataclass(frozen=True)
class Sequence:
    sequence: str
    name: str = field(default='', repr=False)
    extendible: ClassVar[bool] = field(default=False, repr=False) # can't extend simple sequence 

    @property
    def start(self) -> int:
        return 0
    
    @property
    def end(self) -> int:
        return len(self.sequence)   
    
    @property
    def chrom(self) -> str:
        return self.name

    def __len__(self) -> int:
        return len(self.sequence)

    @classmethod
    def zero(cls) -> 'Sequence':
        return cls('')

    def trim(self, left: int, right: int) -> 'GenomeRef':
        return dt_replace(self,
                          sequence= self.sequence[left:-right])
    
    def slice(self, start: int, end: int) -> 'Sequence':
        return dt_replace(sequence=self.sequence[start:end])
    
    def __getitem__(self, item) -> 'str | Sequence':
        if isinstance(item, int):
            return self.sequence[item]
        elif isinstance(item, slice):
            if item.step is not None and  item.step != 1:
                raise IntervalException("Sequence object doesn't support slicing with step different from 1")
            if item.start is None:
                start = 0
            else:
                start = item.start
            end = item.stop
            if start < 0 or end < 0:
                raise IntervalException("Interval object doesn't support slicing with negative indices")
            return self.slice(start, end)
        else:
            raise IntervalException(f"Unsupported indexing item: {item}")

@dataclass(frozen=True)
class CigarPad:
    length: int
    char: str = 'N'
    consumes_ref: ClassVar[bool] = False
    consumes_query: ClassVar[bool] = True # this is different from cigar specification but matches chorus needs more closely
    cigar_symbol: ClassVar[str] = 'P'

    def __len__(self) -> int:
        return self.length

    def split(self, pos: int) -> tuple['CigarPad | None', 'CigarPad | None']:
        if pos <= 0:
            return None, self
        elif pos >= len(self):
            return self, None
        else:
            left = CigarPad(length=pos, char=self.char)
            right = CigarPad(length=self.length - pos, char=self.char)
            return left, right 


@dataclass(frozen=True)
class CigarEqual:
    length: int
    consumes_ref: ClassVar[bool] = True
    consumes_query: ClassVar[bool] = True
    cigar_symbol: ClassVar[str] = '='

    def __len__(self) -> int:
        return self.length

    def split(self, pos: int) -> tuple['CigarEqual | None', 'CigarEqual | None']:
        if pos <= 0:
            return None, self
        elif pos >= len(self):
            return self, None 
        else:
            left = CigarEqual(length=pos)
            right = CigarEqual(length=self.length - pos)
            return left, right 


@dataclass(frozen=True)
class CigarNotEqual:
    seq: str 
    consumes_ref: ClassVar[bool] = True
    consumes_query: ClassVar[bool] = True
    cigar_symbol: ClassVar[str] = 'X'

    def __len__(self) -> int:
        return len(self.seq)

    def split(self, pos: int) ->  tuple['CigarNotEqual | None', 'CigarNotEqual | None']:
        if pos <= 0:
            return None, self
        elif pos >= len(self):
            return self, None 
        else:
            left = CigarNotEqual(seq=self.seq[:pos])
            right = CigarNotEqual(seq=self.seq[pos:])
            return left, right 

@dataclass(frozen=True)
class CigarInsertion:
    seq: str
    consumes_ref: ClassVar[bool] = False
    consumes_query: ClassVar[bool] = True
    cigar_symbol: ClassVar[str] = 'I'


    def __len__(self) -> int:
        return len(self.seq)

    def split(self, pos: int) -> tuple['CigarInsertion | None', 'CigarInsertion | None']:
        if pos <= 0:
            return None, self
        elif pos >= len(self):
            return self, None
        else:
            left = CigarInsertion(seq=self.seq[:pos])
            right = CigarInsertion(seq=self.seq[pos:])
            return left, right 


@dataclass(frozen=True)
class CigarDeletion:
    length: int
    consumes_ref: ClassVar[bool] = True
    consumes_query: ClassVar[bool] = False
    cigar_symbol: ClassVar[str] = 'D'

    def __len__(self) -> int:
        return self.length

    def split(self, pos: int) -> tuple['CigarDeletion | None', 'CigarDeletion | None']:
        if pos <= 0:
            return None, self
        elif pos >= len(self):
            return self, None
        else:
            left = CigarDeletion(length=pos)
            right = CigarDeletion(length=self.length - pos)
            return left, right 



CigarEntry =TypeVar('CigarEntry', 
                     CigarPad, 
                     CigarEqual, 
                     CigarNotEqual, 
                     CigarInsertion, 
                     CigarDeletion)

@dataclass(frozen=True)
class Interval: 
    reference: GenomeRef | Sequence
    cigar: list[CigarEntry]
    hash_sequence: bool = field(default=True, repr=False) # set to False to decrease memory usage

    @classmethod
    def make(cls, reference: GenomeRef | Sequence) -> 'Interval':
        return cls(reference=reference, cigar=[CigarEqual(len(reference))])

    def __post_init__(self):
        object.__setattr__(self, '_sequence', None)

    @property
    def sequence(self) -> str:
        if self._sequence is None:
            if self.hash_sequence:
                sequence = self._get_sequence()
                object.__setattr__(self, '_sequence', sequence)
            else:
                return self._get_sequence()
        return self._sequence

    def _get_sequence(self) -> str:
        reference = self.reference.sequence
        parts = []
        cur_pos = 0
        for entry in self.cigar:
            if isinstance(entry, CigarEqual):
                part = reference[cur_pos:cur_pos+len(entry)]
                parts.append(part)
            elif isinstance(entry, CigarNotEqual):
                parts.append(entry.seq)
            elif isinstance(entry, CigarInsertion):
                parts.append(entry.seq)
            elif isinstance(entry, CigarDeletion):
                pass # deletion does not contribute to the sequence
            elif isinstance(entry, CigarPad):
                part = entry.char * len(entry)
                parts.append(part)
            else:
                raise IntervalException(f"No rules implemented for cigar entry type: {type(entry)}")

            if entry.consumes_ref:
                cur_pos += len(entry)
        return ''.join(parts)

    def cigar_string(self, compressed: bool = True) -> str:
        parts = []

        for entry in self.cigar:
            if compressed:
                part = f"{len(entry)}{entry.cigar_symbol}"
            else:
                part = entry.cigar_symbol * len(entry)
            parts.append(part)

        return ''.join(parts)

    def __len__(self) -> int:
        length = 0
        for entry in self.cigar:
            if entry.consumes_query:
                length += len(entry)
        return length 

    def alignment(self, *,
                  start: int | None = None, 
                  end: int | None = None, # 0-indexed, half-open
                  reference: bool = True) -> Alignment:        
        sequences = [self.reference.sequence, 
                     self.sequence]
        ref_pos = 0
        alt_pos = 0 
        ref_coordinates = [ref_pos]
        alt_coordinates = [alt_pos]
        for entry in self.cigar:
            if entry.consumes_ref:
                ref_pos += len(entry)
            if entry.consumes_query:
                alt_pos += len(entry)
            ref_coordinates.append(ref_pos)
            alt_coordinates.append(alt_pos)

        ali =  Alignment(sequences=sequences, coordinates=np.array([ref_coordinates, alt_coordinates]))
        if start is None and end is None:
            return ali

        if start is None:
            start = 0
        if end is None:
            if reference:
                end = len(sequences[0])
            else:
                end = len(sequences[1])
        if reference:
            si = np.where(ali.indices[0] == start)[0][0]
            ei = np.where(ali.indices[0] == end-1)[0][0] 
        else:
            si = np.where(ali.indices[1] == start)[0][0]
            ei = np.where(ali.indices[1] == end-1)[0][0]
        return ali[:, si:ei]

    def pretty_alignment(self, *, indent: int = 3) -> Alignment:
        ref_start = 0
        for entry in self.cigar:
            if not isinstance(entry, (CigarEqual, CigarPad)):
                break
            if entry.consumes_ref:
                ref_start += len(entry)
        if ref_start == len(self.reference): # all matched:
            reference = self.reference.sequence
            seq = reference[0:indent] + '...' + reference[-indent:]
            ali = Alignment(sequences=[seq, seq], 
                            coordinates=np.array([[0, len(seq)], 
                                                 [0, len(seq)]]
                                                 )
                )
            return ali 

        ref_end = len(self.reference)
        for entry in reversed(self.cigar):
            if not isinstance(entry, (CigarEqual, CigarPad)):
                break
            if entry.consumes_query:
                ref_end -= len(entry)

        ref_start = max(0, ref_start-indent)
        ref_end = min(len(self.reference), ref_end + indent+1)
        return self.alignment(start=ref_start,
                              end=ref_end, 
                              reference=True)

    def ref2query(self, ref_pos: int, mode: str = 'usual', ref_global: bool = False) -> int:
        '''
        Map reference position to query one
        '''
        if ref_global:
            ref_pos -= self.reference.start

        if ref_pos < 0:
            raise IntervalException('Provided position must be non-negative')
        cur_ref_pos = 0
        cur_query_pos = 0 
        for entry in self.cigar:
            if entry.consumes_ref:
                if len(entry) + cur_ref_pos > ref_pos:
                    break
                cur_ref_pos += len(entry)
            if entry.consumes_query:
                cur_query_pos += len(entry)
        else:
            if cur_ref_pos != ref_pos: # end of genome
                raise IntervalException('Provided position is out of reference length')

        # located required entry
        if entry.consumes_query: # there is 1 to 1 correspondense:
            rest = ref_pos - cur_ref_pos
            cur_query_pos += rest 
            return cur_query_pos
        else: # there is no 1 to 1 correspondence:
            if mode == 'right':
                return cur_query_pos # return right query pos
            elif mode == 'left':
                if cur_query_pos == 0:
                    return cur_query_pos
                else:
                    return cur_query_pos - 1
            elif mode == 'exact':
                return -1
            else:
                raise IntervalException(f'Invalid mode value: {mode}')        

    def query2ref(self, query_pos: int, mode: str = 'usual') -> int:
        '''
        Map real position to reference one
        '''
        if query_pos < 0:
            raise IntervalException('Provided position must be non-negative')
        cur_ref_pos = 0
        cur_query_pos = 0 

        for entry in self.cigar:
            if entry.consumes_query:
                if len(entry) + cur_query_pos >= query_pos:
                    break
                cur_query_pos += len(entry)
            if entry.consumes_ref:
                cur_ref_pos += len(entry)

        else:
            if cur_query_pos != query_pos: # end of genome
                raise IntervalException('Provided position is out of reference length')

        # located required entry
        if entry.consumes_ref: # there is 1 to 1 correspondense:
            rest = query_pos - cur_query_pos
            cur_ref_pos += rest 
            return cur_ref_pos
        else: # there is no 1 to 1 correspondence:
            if mode == 'right':
                return cur_ref_pos # return right real pos
            elif mode == 'left':
                if cur_ref_pos == 0:
                    return cur_ref_pos
                else:
                    return cur_ref_pos - 1
            elif mode == 'exact':
                return -1
            else:
                raise IntervalException(f'Invalid mode value: {mode}')   


    def _pos2entry(self, pos: int) -> tuple[int, int]:
        """
        Map position to cigar entry position

        pos: position in the modified sequence 
        """
        if pos < 0:
            raise IntervalException('Provided position must be non-negative')

        cur_pos = 0 
        for ind, entry in enumerate(self.cigar):
            if entry.consumes_query:
                if len(entry) + cur_pos >= pos:
                    pos_in_entry = pos - cur_pos
                    return ind, pos_in_entry 
                else:
                    cur_pos += len(entry)
        if cur_pos == 0 and pos == 0: # cigar contains no entries consuming query
            return 0, 0 # perform operation at the start of the query
        raise IntervalException('Provided position is out of query')

    def insert(self, seq: str, pos: int, align: str = 'usual') -> 'Interval':
        ci = CigarInsertion(seq=seq)      
        entry_pos, split_pos = self._pos2entry(pos=pos)
        new_cigar = self.cigar[:entry_pos]
        left, right = self.cigar[entry_pos].split(split_pos)
        if left is not None:
            new_cigar.append(left)
        ins_pos = len(new_cigar)
        new_cigar.append(ci)
        if right is not None:
            new_cigar.append(right)
        new_cigar.extend(self.cigar[entry_pos+1:])

        interval = Interval(reference=self.reference, cigar=new_cigar).clean()
        if align != 'none':
            interval = interval.realign(ins_pos, extend=True, how=align)
        return interval
        
    def delete(self, start: int, end: int, align: str = 'usual') -> 'Interval':
        left_ref = self.query2ref(query_pos=start, mode='right')
        right_ref = self.query2ref(query_pos=end, mode='right')
        deletion_size = right_ref - left_ref
        if deletion_size > 0:
            cd = CigarDeletion(length=deletion_size)
        else:
            cd = None
        start_entry_pos, start_split_pos = self._pos2entry(pos=start)
        end_entry_pos, end_split_pos = self._pos2entry(pos=end)

        if start_entry_pos != end_entry_pos:
            new_cigar = self.cigar[:start_entry_pos]
            start_entry, _ = self.cigar[start_entry_pos].split(start_split_pos)
            if start_entry is not None:
                new_cigar.append(start_entry)
            del_pos = len(new_cigar)
            if cd is not None:
                new_cigar.append(cd)

            _, end_entry = self.cigar[end_entry_pos].split(end_split_pos)
            if end_entry is not None:
                new_cigar.append(end_entry)
            new_cigar.extend(self.cigar[end_entry_pos+1:])
        else: # start_entry_pos == end_entry_pos
            new_cigar = self.cigar[:start_entry_pos]
            start_entry, end_entry = self.cigar[start_entry_pos].split(start_split_pos)
            if start_entry is not None:
                new_cigar.append(start_entry)
            del_pos = len(new_cigar)
            if cd is not None:
                new_cigar.append(cd)
            end_offset = end_split_pos - start_split_pos
            _, end_entry = end_entry.split(end_offset)
            if end_entry is not None:
                new_cigar.append(end_entry)
            new_cigar.extend(self.cigar[start_entry_pos+1:])

        interval = Interval(reference=self.reference, cigar=new_cigar).clean()
        if align != 'none':
            interval = interval.realign(del_pos, extend=True, how=align)
        return interval

    def _entry2refpos(self, entry_ind: int) -> int:
        ref_pos = 0
        for ind in range(0, entry_ind):
            entry = self.cigar[ind]
            if entry.consumes_ref:
                ref_pos += len(entry)

        return ref_pos

    def realign(self, 
                 initial_start_pos: int = 0,
                 initial_end_pos: int | None = None, # 0-based, half-open
                 extend: bool = True,
                 how='usual') -> 'Interval':
        if initial_end_pos is None:
            initial_end_pos = initial_start_pos + 1

        if extend:
            for start_entry_pos in range(initial_start_pos-1, -1, -1):
                if isinstance(self.cigar[start_entry_pos], (CigarEqual, CigarPad)):
                    start_entry_pos += 1
                    break 
            else: # reached start of cigar
                start_entry_pos = 0

            for end_entry_pos in range(initial_end_pos, len(self.cigar)):
                if isinstance(self.cigar[end_entry_pos], (CigarEqual, CigarPad)):
                    break
            else: # reached end of cigar
                end_entry_pos =  len(self.cigar)
        else:
            start_entry_pos = initial_start_pos
            end_entry_pos = initial_end_pos 
        
        if start_entry_pos +1 == end_entry_pos: # one entry
            return self.copy()

        reference = self.reference.sequence
        cur_pos = self._entry2refpos(entry_ind=start_entry_pos)
        ref_parts = []
        alt_parts = []
        
        for ind in range(start_entry_pos, end_entry_pos):
            entry = self.cigar[ind]
            if isinstance(entry, CigarEqual):
                part = reference[cur_pos:cur_pos+len(entry)]
                ref_parts.append(part)
                alt_parts.append(part)
            elif isinstance(entry, CigarNotEqual):
                part = reference[cur_pos:cur_pos+len(entry)]
                ref_parts.append(part)
                alt_parts.append(entry.seq)
            elif isinstance(entry, CigarInsertion):
                alt_parts.append(entry.seq)
            elif isinstance(entry, CigarDeletion):
                part = reference[cur_pos:cur_pos+len(entry)]
                ref_parts.append(part)
            elif isinstance(entry, CigarPad):
                # very strange case, not sure if we should allow it 
                part = entry.char * len(entry)
                alt_parts.append(part)
            else:
                raise IntervalException(f"No rules implemented for cigar entry type: {type(entry)}")
        
        ref_seq = "".join(ref_parts)
        alt_seq = "".join(alt_parts)

        new_cigar = self.cigar[:start_entry_pos]
        ali = self._pairwise_alignment(target=ref_seq, query=alt_seq, how=how)
        ali_cigar = self._alig2cigar(ali)
        new_cigar.extend(ali_cigar)
        new_cigar.extend(self.cigar[end_entry_pos:])

        return Interval(reference=self.reference, 
                        cigar=new_cigar)

    @staticmethod
    def _alig2cigar(alignment: Alignment) -> list[CigarEntry]:
        target = alignment.target
        query = alignment.query
        cigar= ('I' if i == -1 else 'D' if j == -1 else '=' if target[i] == query[j] else 'X' for i, j in zip(alignment.indices[0], alignment.indices[1]))  
        cigar_groups = []
        querypos = 0
        for g, l in groupby(cigar):
            if g == CigarEqual.cigar_symbol:
                c = CigarEqual(length=ilength(l))
            elif g == CigarInsertion.cigar_symbol:
                c = CigarInsertion(seq=query[querypos:querypos+ilength(l)])
            elif g == CigarDeletion.cigar_symbol:
                c = CigarDeletion(length=ilength(l))
            elif g ==  CigarNotEqual.cigar_symbol: 
                c = CigarNotEqual(seq=query[querypos:querypos+ilength(l)])
            else:
                raise Exception('')
            if c.consumes_query:
                querypos += len(c)
            cigar_groups.append(c)
        return cigar_groups

    @staticmethod
    def _pairwise_alignment(target: str, query: str, how: str = 'usual') -> Alignment:
        sequences=[target, query]
        if len(target) == 0:
            coordinates=np.array([[0, 0],
                                  [0, len(query)] 
                                     ]
                                )
            ali = Alignment(sequences, coordinates=coordinates)
            return ali
        if len(query) == 0:
            coordinates=np.array([[0, len(target)],
                                  [0, 0] ]
                                )
            ali = Alignment(sequences, coordinates=coordinates)
            return ali

        if len(target) == len(query):
            coordinates=np.array([[0, len(target)],
                                      [0, len(query)] 
                                     ]
                                )
            ali = Alignment(sequences, coordinates=coordinates)
            return ali 

        if how == 'left':
            if len(target) > len(query):
                coordinates=np.array([[0, len(query), len(target)],
                                      [0, len(query), len(query)]]
                                    )
            else: #len(target) < len(query):
                coordinates=np.array([[0, len(target), len(target)],
                                      [0, len(target), len(query)] 
                                     ]
                                    )
            ali = Alignment(sequences, coordinates=coordinates)
            return ali 
        elif how == 'right':
            if len(target) > len(query):
                diff = len(target) - len(query)
                coordinates=np.array([[0, diff, len(target)], 
                                      [0, 0, len(query)]]
                                    )
            else: # len(target) < len(query):
                diff = len(query) - len(target)
                coordinates=np.array([[0, 0, len(target)], 
                                      [0, diff, len(query)]]
                                    )
            ali = Alignment(sequences, coordinates=coordinates)
            return ali 

        aligner = Align.PairwiseAligner(scoring="blastn")
        if how == 'usual':
            pass
        else:
            raise IntervalException('Unexpected alignment type')

        alignments = aligner.align(target, query)
        alignment = alignments[0]
        return alignment

    def replace(self, 
                seq: str, 
                start: int, 
                end: int, 
                align: bool = 'usual') -> 'Interval':
        interval = self.delete(start=start, end=end, align='none') 
        # no alignment to avoid any possible coordinate shift
        interval = interval.insert(seq, pos=start, align=align)
        return interval

    def copy(self) -> 'Interval':
        return Interval(reference=self.reference, cigar=self.cigar)

    def extend(self, length: int, how: str = 'both') -> 'Interval':
        if length < len(self):
            return self.copy()
        if self.reference.extendible:            
            extension_needed = length - len(self)
            extended_reference = self.reference.slop(extension_needed=extension_needed, 
                                                     how=how)
            new_cigar = []
            left_equal_length = self.reference.start - extended_reference.start
            if left_equal_length != 0:
                ce = CigarEqual(length=left_equal_length)
                new_cigar.append(ce)
            new_cigar.extend(self.cigar)
            right_equal_length = extended_reference.end - self.reference.end
            if right_equal_length != 0:
                ce = CigarEqual(length=right_equal_length)
                new_cigar.append(ce)
            interval = Interval(reference=extended_reference, 
                                cigar=new_cigar)
            if len(interval) != length:
                interval = interval.pad2length(length=length, how=how)
            return interval.clean()
        else:
            return self.pad2length(length=length, how=how).clean()

    def pad2length(self, length: int, how: str = 'both', char: str='N') -> 'Interval':
        if len(char) != 1:
            raise IntervalException(f'char must be single-char string: {char}')
        required = length - len(self)
        if how == 'left':
            lpad = required
        elif how == 'right':
            lpad = 0            
        elif how == 'both':
            lpad = required // 2
        else:
            raise IntervalException(f"Wrong value of pad how parameter: {how}")
        rpad = required - lpad

        new_cigar = []
        if lpad != 0:
            cp = CigarPad(length=lpad, char=char)
            new_cigar.append(cp)
        new_cigar.extend(self.cigar)
        if rpad != 0:
            cp = CigarPad(length=rpad, char=char)
            new_cigar.append(cp)

        return Interval(reference=self.reference, cigar=new_cigar).clean()

    def clean(self) -> 'Interval':
        new_cigar = [self.cigar[0]]
        for entry in self.cigar[1:]:
            if type(new_cigar[-1]) is type(entry):
                prev_entry = new_cigar.pop()
                if isinstance(entry, CigarEqual):
                    new_entry = CigarEqual(length=len(prev_entry)+ len(entry)) 
                elif isinstance(entry, CigarNotEqual):
                    new_entry = CigarNotEqual(seq=prev_entry.seq + entry.seq)
                elif isinstance(entry, CigarInsertion):
                    new_entry = CigarInsertion(seq=prev_entry.seq + entry.seq)
                elif isinstance(entry, CigarDeletion):
                    new_entry = CigarDeletion(length=len(prev_entry)+ len(entry))
                elif isinstance(entry, CigarPad):
                    if prev_entry.char == entry.char:
                        new_entry = CigarPad(length=len(prev_entry)+ len(entry), char=entry.char)
                    else:
                        new_cigar.append(prev_entry)
                        new_entry = entry 
                else:
                    raise IntervalException(f"No rules implemented for cigar entry type: {type(entry)}")
                new_cigar.append(new_entry)
            else:
                new_cigar.append(entry)
        return Interval(reference=self.reference, cigar=new_cigar)

    def slice(self, start: int, end: int) -> 'Interval':
        start_ref = self.query2ref(start, mode='right')
        end_ref = self.query2ref(end, mode='right')

        start_entry_pos, start_split_pos = self._pos2entry(pos=start)
        end_entry_pos, end_split_pos = self._pos2entry(pos=end)

        new_cigar = []
    
        if start_entry_pos != end_entry_pos:
              
            _, start_entry = self.cigar[start_entry_pos].split(start_split_pos)
            if start_entry is not None:
                new_cigar.append(start_entry)
            new_cigar.extend(self.cigar[start_entry_pos+1:end_entry_pos])
            end_entry, _ = self.cigar[end_entry_pos].split(end_split_pos)
            if end_entry is not None:
                new_cigar.append(end_entry)
        else: # start_entry_pos == end_entry_pos

            _, end_entry = self.cigar[start_entry_pos].split(start_split_pos)
            end_offset = end_split_pos - start_split_pos
            end_entry, _ = end_entry.split(end_offset)
            if end_entry is not None:
                new_cigar.append(end_entry)
    
        new_reference = self.reference.slice(start_ref, end_ref)
       
        return dt_replace(self, cigar=new_cigar, reference=new_reference)

    def __getitem__(self, item) -> 'str | Interval':
        if isinstance(item, int):
            return self.sequence[item]
        elif isinstance(item, slice):
            if item.step is not None and  item.step != 1:
                raise IntervalException("Interval object doesn't support slicing with step different from 1")
            if item.start is None:
                start = 0
            else:
                start = item.start
            end = item.stop
            if start < 0 or end < 0:
                raise IntervalException("Interval object doesn't support slicing with negative indices")
            return self.slice(start, end)
        else:
            raise IntervalException(f"Unsupported indexing item: {item}")

    def prune_reference(self, how='both') -> 'Interval':
        '''
        cut ends of reference not used in the current alignment
        '''
        if how != 'both' and how != 'left' and how != 'right':
            raise IntervalException(f"Wrong how argument: {how}")
        
        new_cigar = copy(self.cigar)
        if how == 'both' or how == 'left':
            for start_ind, entry in enumerate(self.cigar):
                if entry.consumes_query:
                    break
        else:
            start_ind = 0

        if (how == 'both' or how == 'right') and len(new_cigar) > 0:
            for end_ind in range(len(new_cigar)-1, start_ind-1, -1):
                entry = new_cigar[end_ind]
                if entry.consumes_query:
                    end_ind += 1
                    break
        else:
            end_ind = len(new_cigar)

        if start_ind >= end_ind: # no query 
            return Interval.make(self.reference.zero())
        
        cut_left =  sum(len(self.cigar[i]) for i in range(0, start_ind) if self.cigar[i].consumes_ref)
        cut_right = sum(len( self.cigar[i]) for i in range(end_ind, len(self.cigar)) if self.cigar[i].consumes_ref)

        new_cigar = self.cigar[start_ind:end_ind]
        new_reference = self.reference.trim(cut_left, cut_right)
        return dt_replace(self, reference=new_reference, cigar=new_cigar)

    def __radd__(self, other: 'str | Interval') -> 'Interval':
        # This is called if 'other' (e.g., an int) is on the left
        if isinstance(other, str):
            return self.insert(other, 0, align='left')
        else:
            raise IntervalException(f"Unsupported addition of type {type(other)}")

    def __add__(self, other: 'str | Interval') -> 'Interval':
        if isinstance(other, str):
            return self.insert(other, len(self), align='right')
        elif isinstance(other, Interval):
            if type(self.reference) is not type(other.reference):
                raise IntervalException(f"Added intervals must share the same type of reference")
            if isinstance(self.reference, Sequence): # and isinstance(other, Sequence)
                add_reference = Sequence(sequence=self.reference.sequence + other.reference.sequence)
                add_cigar = self.cigar + other.cigar
                conflict_position = len(self.cigar)
                add_interval = Interval(reference=add_reference, cigar=add_cigar, hash_sequence=self.hash_sequence).realign(conflict_position)
                return add_interval
            elif isinstance(self.reference, GenomeRef):# and isinstance(other, GenomeRef)
                if self.reference.fasta != other.reference.fasta:
                    raise IntervalException(f"Added intervals must share the same reference")
                if self.reference.chrom != other.reference.chrom:
                    raise IntervalException(f"Added intervals must share the same reference")
                if isinstance(self.cigar[-1], CigarPad):
                    raise IntervalException(f'Right end of left interval should not be padded')
                if isinstance(other.cigar[0], CigarPad):
                    raise IntervalException(f'Left end of right interval should not be padded')

                self = self.prune_reference(how='right')
                other = other.prune_reference(how='left')

                if len(self) == 0:
                    return other.copy()
                if len(other) == 0:
                    return self.copy()

                if self.reference.end > other.reference.start:
                    raise IntervalException(f"Added intervals must not overlap")
                
                new_cigar = copy(self.cigar)
                if self.reference.end != other.reference.start:
                    cd = CigarDeletion(other.reference.start-self.reference.end)
                    conflict_position = len(new_cigar)
                    new_cigar.append(cd)
                else:
                    conflict_position = None 
                new_cigar.extend(other.cigar)

                new_reference = dt_replace(self.reference, end=other.reference.end)

                add_res = dt_replace(self,
                                     reference=new_reference,
                                     cigar=new_cigar)
                if conflict_position is not None:
                    add_res = add_res.realign(conflict_position)
                else:
                    add_res = add_res.clean()
                return add_res
                

        raise IntervalException(f"Unsupported addition of type {type(other)}")