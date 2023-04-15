import dataclasses
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from pulse_analyzer.driver import run_processing
from pulse_analyzer.tools import DifferenceParameter, PulseParameter
from pulse_analyzer.tools.pulses import find_pulses


argument_parser = ArgumentParser(
    prog='differences',
    description='Analyzes pulse events in image stacks. This tool calculates the differences in location between '
                'consecutive frames. All pixels that had no event in an earlier frame is a heatmap that shows how '
                'many events have occured at a pixel. The output is written as `.png`-image and as `.csv`-files.',
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


difference_parameter = DifferenceParameter()
pulse_parameter = PulseParameter()


def plot_difference_graph(image_stack: ndarray,
                          image_id: str,
                          result_path_stem: Path):

    frames, rows, columns = image_stack.shape
    all_pulse_infos = dict()
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c, **dataclasses.asdict(pulse_parameter))
            for pulse_info in pulse_infos:
                for frame in range(pulse_info.duration):
                    all_pulse_infos[(pulse_info.start + frame, c, r)] = pulse_info

    # Find the maximum pulses per frame
    events_per_frame = [
        sum(
            [
                1 for (f, _, _) in all_pulse_infos.keys()
                if f == frame
            ]
        )
        for frame in range(frames)
    ]
    max_frame_events = max(events_per_frame)

    result = []
    active_set = set()
    for frame in range(0, frames):
        current_set = {(coord[1], coord[2]) for coord in all_pulse_infos.keys() if coord[0] == frame}
        if len(current_set) < max_frame_events * difference_parameter.consider_factor:
            result.append(0)
            continue

        total_event_pixel = len(active_set) + len(current_set)
        non_matching = total_event_pixel - len(active_set.intersection(current_set))
        result.append(non_matching / total_event_pixel) if total_event_pixel != 0 else total_event_pixel
        active_set = current_set

    plt.clf()
    plt.title(image_id)
    plt.xlim(0, 60)
    plt.plot(range(0, 60), result)
    plt.savefig(result_path_stem.with_suffix('.png'))


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    run_processing(
        input_path,
        output_path,
        ['*.tif', '*.tiff'],
        {p.get_name(): dataclasses.asdict(p) for p in [difference_parameter, pulse_parameter]},
        plot_difference_graph
    )


if __name__ == '__main__':
    cli()
