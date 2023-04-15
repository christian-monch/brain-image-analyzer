from __future__ import annotations

import dataclasses
import enum
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Tuple

import numpy as np
import progressbar
from matplotlib import pyplot as plt
from skimage import io
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs



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


def find_pulses_at(image_stack, row, column) -> List[PulseInfo]:
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


def find_all_pulses(image_stack) -> Tuple[dict, Tuple[int, int, int]]:
    rows, columns = image_stack[0].shape
    max_pulses = None
    all_pulses = dict()
    for r in range(rows):
        for c in range(columns):
            pulse_info_list = find_pulses_at(image_stack, r, c)
            pulse_count = len(pulse_info_list)
            if pulse_info_list:
                all_pulses[(c, r)] = pulse_count
            if max_pulses is None or pulse_count > max_pulses[0]:
                max_pulses = (pulse_count, c, r)
    return all_pulses, max_pulses


def get_multiple_event_histogram(pulses, max_pulses) -> List[int]:
    return [pulse_count for pulse_count in pulses.values()]


def test_clustering(image_stack, file_name):

    all_pulses, max_pulses = find_all_pulses_image(image_stack)

    all_pulses_image = np.zeros([512, 512])
    for (row, column), value in all_pulses.items():
        all_pulses_image[row][column] = 255

    #plt.title(file_name)
    #plt.imshow(image_stack[0], vmin=0, vmax=255, cmap='gray', interpolation=None)
    #plt.show()

    #plt.title(file_name)
    #plt.imshow(all_pulses_image, vmin=0, vmax=255, cmap='inferno', interpolation=None, )
    #plt.show()


    # Alpha
    from matplotlib.colors import ListedColormap

    cmap = plt.cm.inferno
    alpha_cmap = cmap(np.arange(cmap.N))
    alpha_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    alpha_cmap = ListedColormap(alpha_cmap)

    plt.title(file_name)
    plt.imshow(image_stack[0], vmin=0, vmax=255, cmap='gray', interpolation=None)
    plt.imshow(all_pulses_image, vmin=0, vmax=255, alpha=0.9, cmap=alpha_cmap, interpolation='bilinear')
    #plt.show()

    #plt.close("all")
    #plt.figure(1)
    #plt.clf()

    all_pulses_array = np.zeros([len(all_pulses), 2], dtype=np.int32)
    for index, (row, column) in enumerate(all_pulses):
        all_pulses_array[index] = [column, row]

    X = all_pulses_array
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    for (x, y) in X:
        plt.plot(x, y, 'r.')
    plt.show()

    #from sklearn.datasets import make_blobs
    #X = make_blobs()[0]
    clustering = AffinityPropagation(max_iter=255).fit(X)
    cluster_centers_indices = clustering.cluster_centers_indices_
    number_of_clusters = len(cluster_centers_indices)
    labels = clustering.labels_

    colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 4)))

    for k, col in zip(range(number_of_clusters), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.scatter(
            X[class_members, 0], X[class_members, 1], color=col["color"], marker="."
        )
        plt.scatter(
            cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
        )
        for x in X[class_members]:
            plt.plot(
                [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"]
            )

    plt.title("Estimated number of clusters: %d" % number_of_clusters)
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.show()

    exit(0)


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    image_paths = [
        p.relative_to(input_path)
        for p in chain(input_path.rglob('*.tif'), input_path.rglob('*.tiff'))
        if p.relative_to(input_path).parts[0] != '.git'
    ]

    max_pulse_count = 0
    bar = progressbar.ProgressBar(max_value=len(image_paths))
    for index, image_path in enumerate(image_paths):

        file_name, image_name = image_path.parts[-2], image_path.parts[-1]

        image_id = file_name + '\n' + image_name
        csv_name = file_name + ' - ' + image_name

        result_path = output_path / "histogram" / image_path.with_suffix('.png')
        result_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = result_path.with_suffix('.csv')

        image_stack = io.imread(input_path / image_path)
        pulses, max_pulses = find_all_pulses(image_stack)
        if len(pulses.values()) == 0:
            # we ignore images without pulses
            print(f'ignoring image stack without pulses: f{input_path / image_path}')
            bar.update(index + 1)
            continue

        max_pulse_count = max([max_pulse_count, max(pulses.values())])

        plt.figure()
        plt.title(image_id)
        plt.ylim(0.0, 1.0)
        results = plt.hist(pulses.values(), align='left', bins=[1, 2, 3, 4, 5, 6, 7], rwidth=0.8, density=True)

        plt.savefig(result_path)
        plt.close()

        total_active_pixels = len(pulses)
        csv_header_lines = [
            f'name,"{csv_name}"',
            f'total active points, {total_active_pixels}']
        csv_data_lines = [
            f'{index + 1}, {value}'
            for index, value in enumerate(results[0])]
        csv_path.write_text('\n'.join(csv_header_lines + csv_data_lines))
        bar.update(index + 1)

    print("Max pulse count:", max_pulse_count)


if __name__ == '__main__':
    cli()
