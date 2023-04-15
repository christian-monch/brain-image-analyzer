from __future__ import annotations

import dataclasses
import enum
from typing import List

from numpy import ndarray


class State(enum.Enum):
    wait_for_lower = 1
    entered_lower = 2
    entered_higher = 3


@dataclasses.dataclass
class PulseInfo:
    start: int
    duration: int = 0
    value: int = 0


def find_pulses(image_stack: ndarray,
                row: int,
                column: int,
                min_difference: int = 50,
                lower_factor: float = 1/3,
                upper_factor: float = 1/3,
                min_duration: int = 2,
                ) -> List[PulseInfo]:

    pixel_values = [
        image[row][column]
        for image in image_stack
    ]

    pulses = []
    lowest = min(pixel_values)
    highest = max(pixel_values)
    if highest - lowest > min_difference:
        lower_threshold = lowest + ((highest - lowest) * lower_factor)
        upper_threshold = highest - ((highest - lowest) * upper_factor)
        state = State.wait_for_lower
        for index, value in enumerate(pixel_values):
            if state == State.wait_for_lower:
                if value <= lower_threshold:
                    state = State.entered_lower
            elif state == State.entered_lower:
                if value >= upper_threshold:
                    state = State.entered_higher
                    pulse_info = PulseInfo(start=index)
                    duration = 1
                    max_value = value
            elif state == State.entered_higher:
                if value < upper_threshold:
                    state = State.wait_for_lower
                    if duration >= min_duration:
                        pulse_info.duration = duration
                        pulse_info.value = max_value
                        pulses.append(pulse_info)
                else:
                    duration += 1
                    max_value = max(value, max_value)
    return pulses


def find_all_pulses(image_stack: ndarray,
                    min_difference: int = 50,
                    lower_factor: float = 1/3,
                    upper_factor: float = 1/3,
                    min_duration: int = 2) -> dict[tuple[int, int, int], PulseInfo]:

    frames, rows, columns = image_stack.shape
    all_pulses = dict()
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c, min_difference, lower_factor, upper_factor, min_duration)
            for pulse_info in pulse_infos:
                for pulse_frame in range(pulse_info.duration):
                    all_pulses[(pulse_info.start + pulse_frame, r, c)] = pulse_info
    return all_pulses
