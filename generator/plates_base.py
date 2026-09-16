from __future__ import annotations

from dataclasses import dataclass

BORDER_INSET = 2.25
BORDER_WIDTH = 4.90
FIELD = BORDER_INSET + BORDER_WIDTH / 2
FIELD_INNER = BORDER_INSET + BORDER_WIDTH
PLATE_RADIUS = 13.0
CELL_RADIUS = 9.5
SMALL_CELL_RADIUS = 7.0


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    height: float
    gap: float = 8.2


@dataclass(frozen=True)
class TextLine:
    segments: tuple[Segment, ...]
    x0: float
    x1: float
    bottom: float
    justify: bool = True


@dataclass(frozen=True)
class Mark:
    name: str
    x0: float
    x1: float
    bottom: float
    height: float


@dataclass(frozen=True)
class Cell:
    x0: float
    y0: float
    x1: float
    y1: float
    radius: float = CELL_RADIUS


@dataclass(frozen=True)
class Divider:
    x: float
    y0: float
    y1: float


@dataclass(frozen=True)
class Geometry:
    width: float
    height: float
    lines: tuple[TextLine, ...]
    cells: tuple[Cell, ...] = ()
    dividers: tuple[Divider, ...] = ()
    marks: tuple[Mark, ...] = ()
    flag: tuple[float, float, float, float] | None = None
    yellow_from: float | None = None
    field_radius: float = CELL_RADIUS
    plate_radius: float = PLATE_RADIUS

    @property
    def field(self) -> Cell:
        return Cell(FIELD, FIELD, self.width - FIELD, self.height - FIELD, self.field_radius)


def line(segments: tuple[Segment, ...], x0: float, x1: float, bottom: float,
         justify: bool = True) -> TextLine:
    return TextLine(segments, x0, x1, bottom, justify)
