from __future__ import annotations

from generator.plates_base import Segment, TextLine, line

LINES: dict[str, tuple[TextLine, ...]] = {
    'type1': (
        line((Segment(0, 1, 58, 6.80), Segment(1, 4, 76, 8.30), Segment(4, 6, 58, 6.80),), 33.60, 373.70, 95.00),
        line((Segment(6, 8, 58, 7.60),), 410.50, 487.90, 71.70, justify=False),
    ),
    'type1@3': (
        line((Segment(0, 1, 58, 6.80), Segment(1, 4, 76, 8.30), Segment(4, 6, 58, 6.80),), 28.30, 359.00, 95.60),
        line((Segment(6, 9, 58, 4.50),), 387.30, 500.00, 71.70, justify=False),
    ),
    'type1a': (
        line((Segment(0, 1, 58, 6.80), Segment(1, 4, 58, 6.80),), 49.30, 241.80, 74.50),
        line((Segment(4, 6, 58, 12.80),), 41.30, 137.30, 154.60, justify=False),
        line((Segment(6, 8, 45, 6.30),), 201.20, 263.20, 139.10, justify=False),
    ),
    'type1a@3': (
        line((Segment(0, 1, 58, 6.80), Segment(1, 4, 58, 6.80),), 49.30, 241.70, 74.50),
        line((Segment(4, 6, 58, 12.80),), 32.70, 128.80, 154.20, justify=False),
        line((Segment(6, 9, 45, 9.50),), 171.20, 264.90, 139.20, justify=False),
    ),
    'type1b': (
        line((Segment(0, 2, 58, 6.80), Segment(2, 5, 76, 8.30),), 53.70, 346.60, 94.30),
        line((Segment(5, 7, 58, 7.90),), 397.70, 474.00, 73.50, justify=False),
    ),
    'type2': (
        line((Segment(0, 2, 58, 6.80), Segment(2, 6, 76, 8.30),), 35.10, 371.10, 94.40),
        line((Segment(6, 8, 58, 7.78),), 412.10, 489.90, 71.20, justify=False),
    ),
    'type3': (
        line((Segment(0, 4, 76, 9.63),), 39.20, 250.20, 93.70, justify=False),
        line((Segment(4, 6, 58, 7.10),), 23.10, 127.30, 188.40, justify=False),
        line((Segment(6, 8, 58, 7.10),), 179.80, 260.50, 171.30, justify=False),
    ),
    'type4': (
        line((Segment(0, 4, 45, 10.70),), 31.70, 161.30, 59.20, justify=False),
        line((Segment(4, 6, 45, 4.10),), 14.50, 89.20, 132.50, justify=False),
        line((Segment(6, 8, 45, 3.05),), 111.90, 171.30, 133.40, justify=False),
    ),
    'type4a': (
        line((Segment(0, 2, 45, 3.90),), 13.60, 88.20, 58.40, justify=False),
        line((Segment(2, 4, 45, 2.88),), 112.30, 169.50, 57.70, justify=False),
        line((Segment(4, 8, 45, 10.70),), 29.60, 158.80, 133.20, justify=False),
    ),
    'type4b': (
        line((Segment(0, 2, 45, 3.88),), 15.30, 94.90, 58.90, justify=False),
        line((Segment(2, 4, 45, 6.75),), 106.90, 169.80, 59.20, justify=False),
        line((Segment(4, 6, 45, 10.90),), 20.05, 87.15, 137.85, justify=False),
        line((Segment(6, 8, 45, 0.60),), 109.36, 170.94, 134.50, justify=False),
    ),
    'type5': (
        line((Segment(0, 4, 76, 8.30), Segment(4, 6, 58, 6.80),), 38.60, 364.20, 96.80),
        line((Segment(6, 8, 58, 4.90),), 393.60, 474.40, 73.60, justify=False),
    ),
    'type6': (
        line((Segment(0, 2, 58, 6.80), Segment(2, 6, 76, 8.30),), 51.73, 371.07, 95.20),
        line((Segment(6, 8, 58, 4.83),), 399.30, 481.70, 72.10, justify=False),
    ),
    'type7': (
        line((Segment(0, 4, 76, 9.90),), 40.80, 247.10, 95.70, justify=False),
        line((Segment(4, 6, 58, 5.50),), 25.20, 128.00, 186.60, justify=False),
        line((Segment(6, 8, 58, 5.92),), 180.70, 258.90, 186.60, justify=False),
    ),
    'type8': (
        line((Segment(0, 4, 45, 10.70),), 32.50, 158.90, 60.70, justify=False),
        line((Segment(4, 6, 45, 4.10),), 16.00, 88.60, 132.00, justify=False),
        line((Segment(6, 8, 45, 3.40),), 110.60, 166.90, 133.30, justify=False),
    ),
    'type9': (
        line((Segment(0, 3, 76, 8.30), Segment(3, 5, 58, 6.80), Segment(5, 6, 76, 8.30),), 35.70, 372.00, 96.00),
        line((Segment(6, 8, 58, 7.90),), 404.00, 481.60, 73.00, justify=False),
    ),
    'type10': (
        line((Segment(0, 3, 76, 8.30), Segment(3, 4, 58, 6.80), Segment(4, 7, 76, 8.30),), 35.20, 372.00, 96.00),
        line((Segment(7, 9, 58, 7.90),), 404.00, 481.60, 73.00, justify=False),
    ),
    'type11': (
        line((Segment(0, 1, 45, 4.00), Segment(1, 4, 45, 4.00),), 26.60, 168.40, 62.20),
        line((Segment(4, 6, 45, 4.40),), 25.50, 83.30, 132.20, justify=False),
        line((Segment(6, 8, 45, 1.90),), 110.00, 165.70, 134.10, justify=False),
    ),
    'type15': (
        line((Segment(0, 2, 58, 6.80), Segment(2, 5, 76, 8.30), Segment(5, 6, 58, 6.80),), 46.10, 363.40, 93.70),
        line((Segment(6, 8, 58, 5.90),), 388.00, 473.10, 71.80, justify=False),
    ),
    'type16': (
        line((Segment(0, 2, 58, 10.30),), 20.30, 128.80, 112.10, justify=False),
        line((Segment(2, 4, 58, 11.05),), 157.90, 242.40, 108.80, justify=False),
        line((Segment(4, 8, 58, 24.00),), 26.30, 232.40, 206.80, justify=False),
    ),
    'type17': (
        line((Segment(0, 2, 58, 10.10),), 20.00, 128.90, 74.00, justify=False),
        line((Segment(2, 4, 58, 11.00),), 157.60, 242.50, 73.70, justify=False),
        line((Segment(4, 8, 58, 24.00),), 26.30, 232.70, 168.20, justify=False),
    ),
    'type18': (
        line((Segment(0, 2, 58, 10.00),), 18.20, 128.00, 70.40, justify=False),
        line((Segment(2, 4, 58, 11.00),), 153.90, 243.20, 70.40, justify=False),
        line((Segment(4, 8, 58, 21.50),), 23.90, 232.50, 207.70, justify=False),
    ),
    'type19': (
        line((Segment(0, 1, 76, 8.30),), 50.10, 58.20, 93.60, justify=False),
        line((Segment(1, 3, 58, 6.80), Segment(3, 6, 76, 8.30),), 107.80, 380.80, 94.70),
        line((Segment(6, 8, 58, 7.40),), 410.30, 488.90, 71.90, justify=False),
    ),
    'type20': (
        line((Segment(0, 1, 58, 6.80), Segment(1, 5, 76, 8.30),), 53.00, 340.90, 98.80),
        line((Segment(5, 7, 58, 4.40),), 384.10, 459.00, 77.20, justify=False),
    ),
    'type21': (
        line((Segment(0, 3, 76, 8.30), Segment(3, 4, 58, 6.80),), 86.16, 335.94, 97.80),
        line((Segment(4, 6, 58, 4.40),), 384.80, 459.00, 76.80, justify=False),
    ),
    'type22': (
        line((Segment(0, 4, 45, 10.70),), 32.10, 159.10, 60.40, justify=False),
        line((Segment(4, 5, 45, 4.00),), 36.60, 71.90, 129.90, justify=False),
        line((Segment(5, 7, 45, 1.70),), 109.90, 165.50, 132.30, justify=False),
    ),
    'type23': (
        line((Segment(0, 1, 76, 8.30),), 24.40, 76.90, 95.30, justify=False),
        line((Segment(1, 3, 58, 6.80), Segment(3, 6, 76, 8.30),), 107.80, 380.30, 94.70),
        line((Segment(6, 8, 58, 7.40),), 409.30, 487.80, 71.90, justify=False),
    ),
    'type24': (
        line((Segment(0, 1, 58, 6.80),), 39.50, 78.70, 71.50, justify=False),
        line((Segment(1, 3, 58, 9.68),), 152.60, 259.10, 72.30, justify=False),
        line((Segment(3, 6, 58, 9.80),), 31.00, 155.60, 153.30, justify=False),
        line((Segment(6, 8, 45, 6.70),), 201.00, 263.10, 138.10, justify=False),
    ),
    'type25': (
        line((Segment(0, 1, 45, 4.00),), 19.50, 50.50, 59.80, justify=False),
        line((Segment(1, 4, 45, 10.50),), 75.90, 172.50, 59.50, justify=False),
        line((Segment(4, 6, 45, 3.90),), 14.80, 89.00, 132.50, justify=False),
        line((Segment(6, 8, 45, 4.80),), 111.80, 169.90, 134.00, justify=False),
    ),
    'type26': (
        line((Segment(0, 1, 76, 8.30),), 24.10, 76.60, 95.30, justify=False),
        line((Segment(1, 3, 58, 6.80), Segment(3, 6, 76, 8.30),), 107.80, 380.80, 94.70),
        line((Segment(6, 8, 58, 7.60),), 410.40, 489.50, 71.90, justify=False),
    ),
    'type27': (
        line((Segment(0, 1, 58, 6.80),), 55.30, 73.00, 75.20, justify=False),
        line((Segment(1, 3, 58, 6.80),), 154.90, 246.00, 76.30, justify=False),
        line((Segment(3, 6, 58, 6.30),), 32.90, 158.10, 152.50, justify=False),
        line((Segment(6, 8, 45, 6.50),), 208.10, 249.10, 137.50, justify=False),
    ),
    'type28': (
        line((Segment(0, 1, 45, 4.00),), 28.50, 40.80, 58.90, justify=False),
        line((Segment(1, 4, 45, 10.50),), 75.80, 172.90, 59.50, justify=False),
        line((Segment(4, 6, 45, 3.90),), 14.60, 89.20, 132.50, justify=False),
        line((Segment(6, 8, 45, 4.70),), 111.20, 170.20, 134.00, justify=False),
    ),
    'type1b@3': (
        line((Segment(0, 2, 58, 6.80), Segment(2, 5, 76, 8.30),), 39.70, 332.60, 94.30),
        line((Segment(5, 8, 58, 4.50),), 381.70, 494.00, 73.50, justify=False),
    ),
}
