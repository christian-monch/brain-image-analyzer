from __future__ import annotations

import dataclasses
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

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

argument_parser.add_argument(
    '-n',
    '--normalize',
    default=None,
    dest='normalize',
    help='if provided, provide a number to which pulses will be normalized.')


pulse_parameter = PulseParameter()

# Pulse number normalization
pulse_value_normalization = True


def get_pulses(image_stack: np.ndarray,
               normalize: int | None):

    frames, rows, columns = image_stack.shape

    # Collect all event pixels
    total_active_points = 0
    events_per_frame: defaultdict[int, list] = defaultdict(list)
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            if pulse_infos:
                total_active_points += 1
            for pulse_info in pulse_infos:
                index = pulse_info.start + int(pulse_info.duration / 2)
                index = frames - 1 if index >= frames else index
                events_per_frame[index].append(pulse_info)

    if normalize:
        max_value = max(*[len(pulse_list) for pulse_list in events_per_frame.values()])
    result = [0] * frames
    for index in range(frames):
        pulse_infos = events_per_frame.get(index, [])
        value = len(pulse_infos)
        result[index] = value * normalize / max_value if normalize else value
    return result, total_active_points


def plot_pulses(image_stack: np.ndarray,
                file_name: str,
                result_path_stem: Path,
                normalize: int | None):

    result_path_stem.parent.mkdir(parents=True, exist_ok=True)

    data, total_active_points = get_pulses(image_stack, normalize)

    plt.clf()
    plt.title(file_name)
    plt.plot(range(image_stack.shape[0]), data)
    plt.savefig(result_path_stem.with_suffix('.png'))

    csv_name = file_name.replace('\n', ' - ')
    csv_header_lines = [
        f'name,"{csv_name}"',
        f'total active points, {total_active_points}',
        'frame, active points' if not normalize else f'frame, active points (normalized to {normalize})']
    csv_data_lines = [f'{frame + 1}, {data}' for frame, data in zip(range(image_stack.shape[0]), data)]
    result_path_stem.with_suffix('.csv').write_text('\n'.join(csv_header_lines + csv_data_lines))


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)
    normalize = None if arguments.normalize is None else int(arguments.normalize)

    run_processing(
        input_path,
        output_path,
        ['*.tif', '*.tiff'],
        {p.get_name(): dataclasses.asdict(p) for p in [pulse_parameter]},
        plot_pulses,
        [normalize]
    )


if __name__ == '__main__':
    cli()
