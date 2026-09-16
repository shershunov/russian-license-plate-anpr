from __future__ import annotations

from generator.plate_frames import CELLS, FLAGS, MARKS
from generator.plate_lines import LINES
from generator.plates_base import (
    BORDER_INSET,
    BORDER_WIDTH,
    CELL_RADIUS,
    FIELD,
    FIELD_INNER,
    PLATE_RADIUS,
    SMALL_CELL_RADIUS,
    Cell,
    Divider,
    Geometry,
    Mark,
    Segment,
    TextLine,
    line,
)

__all__ = ["BORDER_INSET", "BORDER_WIDTH", "CELL_RADIUS", "FIELD", "FIELD_INNER",
           "PLATE_RADIUS",
           "SMALL_CELL_RADIUS", "Cell", "Divider", "Geometry", "Mark", "Segment", "TextLine",
           "line",
           "GEOMETRY", "geometry_for"]

W_LONG, H_LONG = 520.0, 112.0
W_SQUARE, H_SQUARE = 290.0, 170.0
W_TRACTOR, H_TRACTOR = 288.0, 206.0
W_MOTO, H_MOTO = 190.0, 145.0
W_PAPER, H_PAPER = 260.0, 220.0

FLAG_LONG = (471.0, 84.8, 35.8, 14.0)
FLAG_LONG_R3 = (473.1, 84.0, 36.1, 14.5)
FLAG_SQUARE = (245.2, 151.7, 26.6, 9.3)
FLAG_MOTO = (141.4, 71.9, 25.0, 8.6)


def _divider(cell: Cell) -> Divider:
    free_left = abs(cell.x0 - FIELD) > 0.5
    return Divider(cell.x0 if free_left else cell.x1, cell.y0, cell.y1)


def _geometry(key: str, width: float, height: float, yellow_from: float | None = None
              ) -> Geometry:
    cells = CELLS[key]
    dividers: tuple[Divider, ...] = ()
    if (width, height) == (W_MOTO, H_MOTO):
        dividers = tuple(_divider(cell) for cell in cells)
        cells = ()
    return Geometry(
        width=width, height=height, lines=LINES[key], cells=cells, dividers=dividers,
        marks=MARKS[key], flag=FLAGS[key], yellow_from=yellow_from,
    )


SIZES: dict[str, tuple[float, float]] = {
    "type1": (W_LONG, H_LONG), "type1@3": (W_LONG, H_LONG),
    "type1b": (W_LONG, H_LONG), "type1b@3": (W_LONG, H_LONG),
    "type2": (W_LONG, H_LONG), "type5": (W_LONG, H_LONG), "type6": (W_LONG, H_LONG),
    "type9": (W_LONG, H_LONG), "type10": (W_LONG, H_LONG), "type15": (W_LONG, H_LONG),
    "type19": (W_LONG, H_LONG), "type20": (W_LONG, H_LONG), "type21": (W_LONG, H_LONG),
    "type23": (W_LONG, H_LONG), "type26": (W_LONG, H_LONG),
    "type1a": (W_SQUARE, H_SQUARE), "type1a@3": (W_SQUARE, H_SQUARE),
    "type24": (W_SQUARE, H_SQUARE), "type27": (W_SQUARE, H_SQUARE),
    "type3": (W_TRACTOR, H_TRACTOR), "type7": (W_TRACTOR, H_TRACTOR),
    "type4": (W_MOTO, H_MOTO), "type4a": (W_MOTO, H_MOTO), "type4b": (W_MOTO, H_MOTO),
    "type8": (W_MOTO, H_MOTO), "type11": (W_MOTO, H_MOTO), "type22": (W_MOTO, H_MOTO),
    "type25": (W_MOTO, H_MOTO), "type28": (W_MOTO, H_MOTO),
    "type16": (W_PAPER, H_PAPER), "type17": (W_PAPER, H_PAPER), "type18": (W_PAPER, H_PAPER),
}

YELLOW_ZONE = {"type15": 390.0}

GEOMETRY: dict[str, Geometry] = {
    key: _geometry(key, *size, YELLOW_ZONE.get(key))
    for key, size in SIZES.items()
}


def geometry_for(subtype: str, region_length: int, profile: str = "gost") -> Geometry:
    if profile == "competition" and subtype == "type1b":
        base = GEOMETRY["type1@3" if region_length == 3 else "type1"]
        return Geometry(base.width, base.height, base.lines, base.cells, base.marks,
                        None, 390.0, base.field_radius, base.plate_radius)
    if region_length == 3:
        wide = GEOMETRY.get(f"{subtype}@3")
        if wide is not None:
            return wide
    geometry = GEOMETRY.get(subtype)
    if geometry is None:
        raise KeyError(f"No plate geometry for {subtype}")
    return geometry
