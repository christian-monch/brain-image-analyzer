import dataclasses
import enum
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.restoration import denoise_tv_chambolle


min_difference = 50
lower_factor = 1/3
upper_factor = 1/3
min_duration = 2


class State(enum.Enum):
    wait_for_lower = 1
    entered_lower = 2
    entered_higher = 3


@dataclasses.dataclass
class PulseInfo:
    start: int
    duration: int = 0
    value: int = 0


def find_pulses(image_stack, row, column) -> List[PulseInfo]:
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

        #if True: # row == 123 and column == 510:
        #    print(pulses, highest, lowest, lower_threshold, upper_threshold, pixel_values)

    #print(pixel_values)
    #print(pulses)
    return pulses


def find_all_pulses_image(image_stack):
    rows, columns = image_stack[0].shape
    result_image = np.zeros([rows, columns], dtype=np.uint8)
    max_pulses = None
    for r in range(rows):
        for c in range(columns):
            pulse_info = find_pulses(image_stack, r, c)
            pulses = len(pulse_info)
            result_image[r][c] = pulses
            if max_pulses is None or pulses > max_pulses[0]:
                max_pulses = (pulses, r, c)

    print("Max pulses:", max_pulses)
    result_image = result_image * (255 / max_pulses[0])
    print(result_image)
    return result_image, max_pulses


def plot_pulses(image_stack):
    rows, columns = image_stack[0].shape
    result = [0] * image_stack.shape[0]
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            for pulse_info in pulse_infos:
                index = pulse_info.start + int(pulse_info.duration / 2)
                if index >= len(result):
                    index = len(result) - 1
                result[index] += 1
    return result
    #return [v * (100.0 / max_value) if max_value > 0 else 0 for v in result]


def plot_pulses_percent(image_stack):
    rows, columns = image_stack[0].shape
    result = [0] * image_stack.shape[0]

    # Collect all bursting pixels
    bursting_pixels = dict()
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            if pulse_infos:
                bursting_pixels[(r, c)] = pulse_infos
    total_pixel_number = len(bursting_pixels)
    print(total_pixel_number)

    # Find bursts
    burst_per_frame = defaultdict(list)
    for r in range(rows):
        for c in range(columns):
            pulse_infos = bursting_pixels.get((r, c), [])
            for pulse_info in pulse_infos:
                index = pulse_info.start + int(pulse_info.duration / 2)
                if index >= len(result):
                    index = len(result) - 1
                burst_per_frame[index].append(pulse_info)
            if pulse_infos:
                print(r, c, pulse_infos)

    result = [0] * image_stack.shape[0]
    for index, bursts in burst_per_frame.items():
        result[index] = len(bursts) * 100 / total_pixel_number
    return result


def cli():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    image_paths = [
        p.relative_to(input_path)
        for p in input_path.rglob('*.tif')
        if p.relative_to(input_path).parts[0] != '.git'
    ]

    for image_path in image_paths:

        image_name = image_path.parts[-2] + "\n" + image_path.parts[-1]
        result_path = output_path / image_path.with_suffix('.png')
        result_path.parent.mkdir(parents=True, exist_ok=True)

        image_stack = io.imread(input_path / image_path)
        data = plot_pulses(image_stack)

        plt.figure()
        plt.title(image_name)
        plt.plot(range(image_stack.shape[0]), data)
        plt.savefig(result_path)
        plt.close()


if __name__ == '__main__':
    cli()
