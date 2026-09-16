from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ALPHABET = "ABEKMHOPCTYX"
PALETTES = {
    "white": ((0.84, 0.85, 0.81), (0.009, 0.010, 0.009)),
    "yellow": ((0.96, 0.66, 0.025), (0.008, 0.009, 0.008)),
    "black": ((0.012, 0.014, 0.013), (0.83, 0.85, 0.81)),
    "red": ((0.55, 0.012, 0.022), (0.86, 0.85, 0.80)),
    "blue": ((0.012, 0.085, 0.37), (0.88, 0.88, 0.83)),
}


@dataclass(frozen=True)
class PlateSpec:
    subtype: str
    title: str
    width_mm: float
    height_mm: float
    layout: str
    pattern: str
    palette: str = "white"
    material: str = "metal"
    flag: bool = True
    rus: bool = True
    figure: str = ""

    @property
    def plate_type(self) -> str:
        return self.subtype if self.subtype in {"type1", "type1a", "type1b"} else "other"

    @property
    def allows_three_digit_region(self) -> bool:
        return self.subtype in {"type1", "type1a"}


SPECS = (
    PlateSpec("type1", "Легковые, грузовые, автобусы", 520, 112, "car", "LNNNLL", figure="А.1–А.2"),
    PlateSpec("type1a", "Нестандартное место крепления", 290, 170, "square", "LNNNLL", figure="А.3–А.4"),
    PlateSpec("type1b", "Такси и пассажирский транспорт", 520, 112, "taxi", "LLNNN", "yellow", flag=False,
              figure="А.5"),
    PlateSpec("type2", "Прицепы", 520, 112, "long", "LLNNNN", figure="А.6"),
    PlateSpec("type3", "Тракторы и самоходные машины", 288, 206, "tractor", "NNNNLL", figure="А.7"),
    PlateSpec("type4", "Мотоциклы", 190, 145, "moto", "NNNNLL", figure="А.8"),
    PlateSpec("type4a", "Внедорожная мототехника", 190, 145, "atv", "LLNNNN", figure="А.9"),
    PlateSpec("type4b", "Мопеды", 190, 145, "moped", "LLNNLL", figure="А.10"),
    PlateSpec("type5", "Военные автомобили", 520, 112, "long", "NNNNLL", "black", flag=False, figure="А.11"),
    PlateSpec("type6", "Военные прицепы", 520, 112, "long", "LLNNNN", "black", flag=False, figure="А.12"),
    PlateSpec("type7", "Военные тракторы", 288, 206, "tractor", "NNNNLL", "black", flag=False, rus=False,
              figure="А.13"),
    PlateSpec("type8", "Военные мотоциклы", 190, 145, "moto", "NNNNLL", "black", flag=False, figure="А.14"),
    PlateSpec("type9", "Главы дипломатических представительств", 520, 112, "diplomat", "DDDCdN", "red", flag=False,
              figure="А.15"),
    PlateSpec("type10", "Дипломатический транспорт", 520, 112, "diplomat", "DDDsNNN", "red", flag=False, figure="А.16"),
    PlateSpec("type11", "Дипломатические мотоциклы", 190, 145, "diplomat_moto", "sDDDNN", "red", flag=False,
              figure="А.17"),
    PlateSpec("type15", "Ламинированный транзит", 520, 112, "long", "LLNNNL", material="laminate", figure="А.18"),
    PlateSpec("type16", "Бумажный транзит мотоциклов", 260, 220, "paper_top", "LLNNNN", material="paper", flag=False,
              figure="А.19"),
    PlateSpec("type17", "Бумажный транзит военной техники", 260, 220, "paper_bottom", "LLNNNN", material="paper",
              flag=False, figure="А.20"),
    PlateSpec("type18", "Бумажный транзит самоходной техники", 260, 220, "paper_middle", "LLNNNN", material="paper",
              flag=False, figure="А.21"),
    PlateSpec("type19", "Вывоз за пределы страны", 520, 112, "prefix_long", "tLLNNN", figure="А.22"),
    PlateSpec("type20", "Полиция: автомобили", 520, 112, "long", "LNNNN", "blue", flag=False, figure="А.23"),
    PlateSpec("type21", "Полиция: прицепы", 520, 112, "long", "NNNL", "blue", flag=False, figure="А.24"),
    PlateSpec("type22", "Полиция: мотоциклы", 190, 145, "moto", "NNNNL", "blue", flag=False, figure="А.25"),
    PlateSpec("type23", "Классические автомобили", 520, 112, "prefix_long", "kLLNNN", figure="А.26"),
    PlateSpec("type24", "Классические автомобили: квадрат", 290, 170, "prefix_square", "kLLNNN", figure="А.27"),
    PlateSpec("type25", "Классические мотоциклы", 190, 145, "prefix_moto", "kNNNLL", figure="А.28"),
    PlateSpec("type26", "Спортивные автомобили", 520, 112, "prefix_long", "cLLNNN", figure="А.29"),
    PlateSpec("type27", "Спортивные автомобили: квадрат", 290, 170, "prefix_square", "cLLNNN", figure="А.30"),
    PlateSpec("type28", "Спортивные мотоциклы", 190, 145, "prefix_moto", "cNNNLL", figure="А.31"),
)
CATALOG = {spec.subtype: spec for spec in SPECS}


@dataclass(frozen=True)
class Identity:
    subtype: str
    serial: str
    region: str
    text: str
    profile: str

    def to_dict(self) -> dict:
        return asdict(self)


def make_identity(
        subtype: str, region: str, rng: np.random.Generator, profile: str = "gost"
) -> Identity:
    spec = CATALOG[subtype]
    if len(region) == 3 and not spec.allows_three_digit_region:
        if not (profile == "competition" and subtype == "type1b"):
            raise ValueError(f"{subtype} requires a two-digit region in the selected profile")
    pattern = "LNNNLL" if profile == "competition" and subtype == "type1b" else spec.pattern
    fixed = {"d": "D", "s": str(rng.choice(["D", "T"])), "t": "T", "k": "K", "c": "C"}
    serial = "".join(
        str(rng.choice(list(ALPHABET))) if char == "L"
        else str(rng.integers(0, 10)) if char in {"N", "D"}
        else fixed.get(char, char)
        for char in pattern
    )
    if set(serial) == {"0"}:
        serial = serial[:-1] + "1"
    if spec.layout in {"atv", "paper_top", "paper_bottom", "paper_middle"}:
        text = serial[:2] + region + serial[2:]
    else:
        text = serial + region
    return Identity(subtype, serial, region, text, profile)
