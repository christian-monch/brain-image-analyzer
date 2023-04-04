import sys
import enum

import numpy as np
from skimage import io
from matplotlib import pyplot as plt


min_difference = 10
lower_factor = 1/2
upper_factor = 1/2
min_duration = 2


class State(enum.Enum):
    wait_for_lower = 1
    entered_lower = 2
    entered_higher = 3


def find_pulses(image_stack, row, column):
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
                    enter_index = index
                    duration = 1
                    max_value = value
            elif state == State.entered_higher:
                if value < upper_threshold:
                    state = State.wait_for_lower
                    if duration >= min_duration:
                        pulses.append((enter_index, duration, max_value))
                else:
                    duration += 1
                    max_value = max(value, max_value)

        #if True: # row == 123 and column == 510:
        #    print(pulses, highest, lowest, lower_threshold, upper_threshold, pixel_values)

    #print(pixel_values)
    #print(pulses)
    return pulses


def find_all_pulses(image_stack):
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


def main(file_name: str):
    image_stack = io.imread(file_name)
    result_image, max_pulses = find_all_pulses(image_stack)
    plt.imshow(result_image, cmap='gray', vmin=0, vmax=10, interpolation=None)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
