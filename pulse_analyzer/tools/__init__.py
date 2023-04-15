from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class PulseParameter:
    min_difference: int = 50
    lower_factor: float = 1/3
    upper_factor: float = 1/3
    min_duration: int = 2

    @classmethod
    def get_name(cls) -> str:
        return 'pulse'


@dataclasses.dataclass
class DifferenceParameter:
    consider_factor: float = 1/4

    @classmethod
    def get_name(cls) -> str:
        return 'difference'


@dataclasses.dataclass
class ClusterParameter:
    normalize: float | None = 100
    pulse_value_normalize: bool = True

    @classmethod
    def get_name(cls) -> str:
        return 'cluster'


@dataclasses.dataclass
class TileParameter:
    number_of_tiles: int = 8

    @classmethod
    def get_name(cls) -> str:
        return 'tiles'
