from typing import Dict, List, Tuple

import torch
from torch import Tensor

PAD_TOKEN: str = '<pad>'
BOS_TOKEN: str = '<bos>'
EOS_TOKEN: str = '<eos>'
UNREADABLE_TOKEN: str = '#'

DIGITS: str = '0123456789'
GOST_LETTERS: str = 'ABEKMHOPCTYX'
DIPLOMATIC_LETTERS: str = 'D'
LETTERS: str = GOST_LETTERS + DIPLOMATIC_LETTERS

alphabet: List[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNREADABLE_TOKEN] + list(DIGITS) + list(LETTERS)

char_to_idx: Dict[str, int] = {char: idx for idx, char in enumerate(alphabet)}
idx_to_char: Dict[int, str] = {idx: char for idx, char in enumerate(alphabet)}

PAD_IDX: int = char_to_idx[PAD_TOKEN]
BOS_IDX: int = char_to_idx[BOS_TOKEN]
EOS_IDX: int = char_to_idx[EOS_TOKEN]
UNREADABLE_IDX: int = char_to_idx[UNREADABLE_TOKEN]

num_classes: int = len(alphabet)

MAX_PLATE_LEN: int = 9
MAX_SEQ_LEN: int = MAX_PLATE_LEN + 1

UNKNOWN_SUBTYPE: str = 'unknown'

SUBTYPES: Tuple[str, ...] = (
    'type1', 'type1a', 'type1b', 'type2', 'type3', 'type4', 'type4a', 'type4b',
    'type5', 'type6', 'type7', 'type8', 'type9', 'type10', 'type11',
    'type15', 'type16', 'type17', 'type18', 'type19', 'type20', 'type21', 'type22',
    'type23', 'type24', 'type25', 'type26', 'type27', 'type28',
    UNKNOWN_SUBTYPE,
)

subtype_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(SUBTYPES)}
idx_to_subtype: Dict[int, str] = {idx: name for idx, name in enumerate(SUBTYPES)}
num_subtypes: int = len(SUBTYPES)
UNKNOWN_SUBTYPE_IDX: int = subtype_to_idx[UNKNOWN_SUBTYPE]

SLOT_CHARS: Dict[str, str] = {
    'L': GOST_LETTERS,
    'D': DIGITS,
    's': 'DT',
    'd': 'D',
    't': 'T',
    'k': 'K',
    'c': 'C',
    'C': 'C',
}

SUBTYPE_PATTERNS: Dict[str, str] = {
    'type1': 'LDDDLLDDD',
    'type1a': 'LDDDLLDDD',
    'type1b': 'LLDDDDD',
    'type2': 'LLDDDDDD',
    'type3': 'DDDDLLDD',
    'type4': 'DDDDLLDD',
    'type4a': 'LLDDDDDD',
    'type4b': 'LLDDLLDD',
    'type5': 'DDDDLLDD',
    'type6': 'LLDDDDDD',
    'type7': 'DDDDLLDD',
    'type8': 'DDDDLLDD',
    'type9': 'DDDCdDDD',
    'type10': 'DDDsDDDDD',
    'type11': 'sDDDDDDD',
    'type15': 'LLDDDLDD',
    'type16': 'LLDDDDDD',
    'type17': 'LLDDDDDD',
    'type18': 'LLDDDDDD',
    'type19': 'tLLDDDDD',
    'type20': 'LDDDDDD',
    'type21': 'DDDLDD',
    'type22': 'DDDDLDD',
    'type23': 'kLLDDDDD',
    'type24': 'kLLDDDDD',
    'type25': 'kDDDLLDD',
    'type26': 'cLLDDDDD',
    'type27': 'cLLDDDDD',
    'type28': 'cDDDLLDD',
    UNKNOWN_SUBTYPE: '',
}

THREE_DIGIT_REGION_SUBTYPES: Tuple[str, ...] = ('type1', 'type1a')

REGION_START: Dict[str, int] = {name: 6 for name in THREE_DIGIT_REGION_SUBTYPES}

SUBTYPE_MIN_LEN: Dict[str, int] = {
    name: len(pattern) - (1 if name in THREE_DIGIT_REGION_SUBTYPES else 0)
    for name, pattern in SUBTYPE_PATTERNS.items()
}

_SLOT_IDS: Dict[str, List[int]] = {
    slot: [char_to_idx[c] for c in chars] for slot, chars in SLOT_CHARS.items()
}
_DIGIT_IDS: List[int] = [char_to_idx[c] for c in DIGITS]
REGION3_PREFIX: str = '123456789'
REGION3_PREFIX_IDS: List[int] = [char_to_idx[c] for c in REGION3_PREFIX]

READABLE_SUBTYPE_IDS: Tuple[int, ...] = tuple(
    idx for name, idx in subtype_to_idx.items() if SUBTYPE_PATTERNS[name]
)


def is_readable_subtype(subtype: int) -> bool:
    return bool(SUBTYPE_PATTERNS[idx_to_subtype[subtype]])


def _pattern_mask(pattern: str, min_len: int) -> Tensor:
    masks: Tensor = torch.zeros(MAX_SEQ_LEN, num_classes, dtype=torch.bool)
    max_len: int = len(pattern)
    for pos in range(MAX_SEQ_LEN):
        if pos < max_len:
            masks[pos, _SLOT_IDS[pattern[pos]]] = True
            masks[pos, UNREADABLE_IDX] = True
        if min_len <= pos <= max_len:
            masks[pos, EOS_IDX] = True
    return masks


def build_position_masks() -> Tensor:
    return torch.stack([
        _pattern_mask(SUBTYPE_PATTERNS[name], SUBTYPE_MIN_LEN[name]) for name in SUBTYPES
    ])


def build_union_position_mask() -> Tensor:
    return build_position_masks()[list(READABLE_SUBTYPE_IDS)].any(dim=0)


def build_region_gate() -> Tuple[Tensor, Tensor]:
    region_pos: Tensor = torch.full((num_subtypes,), -1, dtype=torch.long)
    max_len: Tensor = torch.zeros(num_subtypes, dtype=torch.long)
    for name, idx in subtype_to_idx.items():
        if name in REGION_START:
            region_pos[idx] = REGION_START[name]
            max_len[idx] = len(SUBTYPE_PATTERNS[name])
    return region_pos, max_len


def encode_plate(text: str) -> List[int]:
    return [char_to_idx[char] for char in text]


def decode_tokens(token_ids: Tensor) -> str:
    chars: List[str] = []
    for token in token_ids.tolist():
        if token in (EOS_IDX, PAD_IDX):
            break
        if token == BOS_IDX:
            continue
        chars.append(idx_to_char[token])
    return ''.join(chars)


def infer_subtype(text: str, candidates: Tuple[str, ...] = SUBTYPES) -> int:
    for name in candidates:
        if SUBTYPE_PATTERNS[name] and is_valid_plate(text, name):
            return subtype_to_idx[name]
    return UNKNOWN_SUBTYPE_IDX


def is_valid_plate(text: str, subtype: str) -> bool:
    pattern: str = SUBTYPE_PATTERNS.get(subtype, '')
    if not pattern:
        return text == ''
    if not SUBTYPE_MIN_LEN[subtype] <= len(text) <= len(pattern):
        return False
    for char, slot in zip(text, pattern):
        if char != UNREADABLE_TOKEN and char not in SLOT_CHARS[slot]:
            return False
    if len(text) == len(pattern) and subtype in REGION_START:
        start: int = REGION_START[subtype]
        if text[start] not in REGION3_PREFIX and text[start] != UNREADABLE_TOKEN:
            return False
    return True
