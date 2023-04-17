import dataclasses
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from pulse_analyzer.driver import run_processing
from pulse_analyzer.tools import PulseParameter
from pulse_analyzer.tools.pulses import find_pulses


argument_parser = ArgumentParser(
    prog='pulse_analyzer',
    description='Analyzes pulse events in image stacks. Output is twofold, `.csv`-files with the results and '
                '`.png`-images with plots of the results.',
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


# Pulse number normalization
pulse_value_normalization = True


def plot_pulses_percent(image_stack: np.ndarray,
                        file_name: str,
                        result_path_stem: Path,
                        show_images=True,
                        normalize: int | None = 100):

    frames, rows, columns = image_stack.shape
    result = [0] * frames

    # Collect all pulsating pixels
    pulse_pixels = dict()
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            if pulse_infos:
                pulse_pixels[(r, c)] = pulse_infos
    total_pixel_number = len(pulse_pixels)

    # Find pulses
    events_per_frame = defaultdict(list)
    for r in range(rows):
        for c in range(columns):
            pulse_infos = pulse_pixels.get((r, c), [])
            for pulse_info in pulse_infos:
                index = pulse_info.start + int(pulse_info.duration / 2)
                if index >= len(result):
                    index = len(result) - 1
                events_per_frame[index].append(pulse_info)

    result = [0] * image_stack.shape[0]
    for index, bursts in events_per_frame.items():
        result[index] = len(bursts) * 100 / total_pixel_number
    return result


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    run_processing(
        input_path,
        output_path,
        ['*.tif', '*.tiff'],
        {p.get_name(): dataclasses.asdict(p) for p in [PulseParameter()]},
        plot_pulses_percent
    )


if __name__ == '__main__':
    cli()
