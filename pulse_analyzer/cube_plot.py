import dataclasses
import enum
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import io

from pulse_analyzer.driver import run_processing


argument_parser = ArgumentParser(
    prog='cube_plot',
    description='Analyzes pulse events in image stacks. Outputs are 3D scatter plots of all events, where the x-axis '
                'are the frames.',
    epilog='for more information see: https://github.com/christian-monch/brain-image-analyzer')

argument_parser.add_argument(
    'image_path',
    help='directory that contains image stacks as `.tif` or `.tiff` files')

argument_parser.add_argument(
    'output_path',
    help='directory that will contain outputs, existing files will be overriden, if the directory does not exists, it '
         'will be created')


cmap = plt.cm.inferno
alpha_cmap = cmap(np.arange(cmap.N))
alpha_cmap[:, -1] = np.linspace(0, 1, cmap.N)
alpha_cmap = ListedColormap(alpha_cmap)


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


def row_col_2_plot_space(row, column, height=512) -> Tuple[int, int]:
    return height - 1 - row, column


def x_y_2_row_column(row, column, height=512) -> Tuple[int, int]:
    return column, 512 - row


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
    return pulses


def all_pulses(image_stack):
    rows, columns = image_stack[0].shape
    result = dict()
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            for pulse_info in pulse_infos:
                for duration_index in range(pulse_info.duration):
                    result[(r, c, pulse_info.start + duration_index)] = pulse_info.value
    return result


def plot_events(image_stack,
                image_id: str,
                result_path: Path):

    cube = all_pulses(image_stack)

    plt.title(image_id)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for (x, y, z), value in cube.items():
        ax.scatter(z, x, y, marker='.')

    ax.set_xlabel('Frame')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 512)
    ax.set_zlim(0, 512)

    print(f"starting plot")
    plt.savefig(result_path)
    plt.close(fig)


def cube_plot(input_path, output_path, image_path):

    file_name, image_name = image_path.parts[-2], image_path.parts[-1]
    image_id = file_name + '\n' + image_name

    result_path = output_path / image_path.with_suffix('.png')
    result_path.parent.mkdir(parents=True, exist_ok=True)

    image_stack = io.imread(input_path / image_path)
    plot_events(image_stack, image_id, result_path)


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    run_processing(input_path, output_path, ['*.tif', '*.tiff'], cube_plot)


if __name__ == '__main__':
    cli()
