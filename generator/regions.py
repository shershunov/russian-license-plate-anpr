from __future__ import annotations

EXTRA_SERIES = (
    "102", "103", "113", "116", "121", "123", "124", "125", "126", "134", "136",
    "138", "142", "147", "150", "152", "154", "155", "159", "161", "163", "164",
    "173", "174", "177", "178", "186", "190", "193", "196", "197", "198", "199",
    "224", "250", "252", "277", "299", "323", "550", "702", "716", "725", "750",
    "761", "763", "774", "777", "778", "790", "797", "799", "977",
)
BASE_CODES = tuple(f"{number:02d}" for number in range(1, 100))


def resolve_regions(requested: tuple[str, ...], profile: str) -> tuple[str, ...]:
    if requested == ("catalog",):
        codes = BASE_CODES + EXTRA_SERIES
    elif requested == ("base",):
        codes = BASE_CODES
    elif requested == ("code_space",):
        prefixes = "127" if profile == "competition" else "123456789"
        codes = BASE_CODES + tuple(prefix + code for prefix in prefixes for code in BASE_CODES)
    else:
        codes = requested
    if not codes:
        raise ValueError("At least one region is required")
    for code in codes:
        if not isinstance(code, str) or not code.isascii() or not code.isdigit():
            raise ValueError(f"Invalid region: {code!r}")
        if len(code) not in {2, 3} or int(code) == 0 or (len(code) == 3 and code[0] == "0"):
            raise ValueError(f"Invalid region: {code}")
    if profile == "competition":
        codes = tuple(code for code in codes if len(code) == 2 or code[0] in "127")
    if not codes:
        raise ValueError("No requested region fits the selected profile")
    return tuple(dict.fromkeys(codes))
