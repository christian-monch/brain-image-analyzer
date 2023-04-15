import dataclasses
import enum
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import io
from sklearn.cluster import DBSCAN

from pulse_analyzer.driver import run_processing


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


# Pulse finding parameter
min_difference = 50
lower_factor = 1/3
upper_factor = 1/3
min_duration = 2


# Pulse number normalization
pulse_value_normalization = True


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


def find_all_pulses_image(image_stack):
    rows, columns = image_stack[0].shape
    result_image = np.zeros([rows, columns], dtype=np.uint8)
    max_pulses = None
    all_pulses = dict()
    for r in range(rows):
        for c in range(columns):
            pulse_info = find_pulses(image_stack, r, c)
            pulses = len(pulse_info)
            result_image[r][c] = pulses
            if pulse_info:
                all_pulses[(r, c)] = pulses
            if max_pulses is None or pulses > max_pulses[0]:
                max_pulses = (pulses, r, c)

    if pulse_value_normalization is True:
        result_image = result_image * ((255 / max_pulses[0]) if max_pulses[0] != 0 else 1)
    return result_image, all_pulses, max_pulses


def get_pulses(image_stack):
    rows, columns = image_stack[0].shape
    result = [0] * image_stack.shape[0]

    # Collect all pulsating pixels
    pulsing_pixels = dict()
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            if pulse_infos:
                pulsing_pixels[(r, c)] = pulse_infos
    total_pixel_number = len(pulsing_pixels)

    # sum up pulses per pixel
    for r in range(rows):
        for c in range(columns):
            pulse_infos = pulsing_pixels.get((r, c), [])
            for pulse_info in pulse_infos:
                index = pulse_info.start + int(pulse_info.duration / 2)
                if index >= len(result):
                    index = len(result) - 1
                result[index] += 1
    return total_pixel_number, result


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
    #print(total_pixel_number)

    # Find pulses
    events_per_frame = defaultdict(list)
    for r in range(rows):
        for c in range(columns):
            pulse_infos = bursting_pixels.get((r, c), [])
            for pulse_info in pulse_infos:
                index = pulse_info.start + int(pulse_info.duration / 2)
                if index >= len(result):
                    index = len(result) - 1
                events_per_frame[index].append(pulse_info)

    result = [0] * image_stack.shape[0]
    for index, bursts in events_per_frame.items():
        result[index] = len(bursts) * 100 / total_pixel_number
    return result


def xxx(image_stack, file_name, result_path, csv_id, csv_path, show_images=True, normalize=100):

    _, all_pulses, max_pulses = find_all_pulses_image(image_stack)

    all_pulses_image = np.zeros([512, 512])
    for (row, column), value in all_pulses.items():
        all_pulses_image[row][column] = value

    X = [(column, row) for (row, column) in all_pulses]
    if not X:
        print(f"Ignoring {file_name} because no events were found")
        return

    clustering = DBSCAN(eps=5).fit(X)
    labels = clustering.labels_
    number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    clusters = [
        [(x, y) for label, (x, y) in zip(clustering.labels_, X) if label == outer_label]
        for outer_label in set(labels)
        if outer_label != -1
    ]

    if show_images is True:
        plt.clf()
        plt.xlim(0, 512)
        plt.ylim(0, 512)
        plt.title(file_name + f' ({number_of_clusters} clusters)')
        plt.imshow(image_stack[0], vmin=0, vmax=255, cmap='gray', interpolation=None)
        plt.imshow(all_pulses_image, vmin=0, vmax=255, alpha=0.9, cmap=alpha_cmap, interpolation='bilinear')

        # Plot detected pulses
        for (x, y) in X:
            plt.plot(x, y, 'r.')

        # Plot clusters
        for index, cluster in enumerate(clusters):
            for (x, y) in cluster:
                plt.plot(x, y, f'C{index % 10}.')

        #plt.savefig(result_path.parent / (result_path.stem + "-cluster" + result_path.suffix))
        plt.show()

    # Determine the events per cluster per time frame
    # Those have already been calculated, but we did not store them yet!

    # Go through all clusters
    result_stack = []
    for cluster in clusters:

        # Go through all pixel of the cluster and record the number of events
        # in a given frame
        cluster_result = 60 * [0]
        for (x, y) in cluster:
            pulse_infos = find_pulses(image_stack, y, x)
            for pulse_info in pulse_infos:
                index = int(pulse_info.start + (pulse_info.duration + .5) / 2)
                index = index if index < 60 else 59
                cluster_result[index] += 1

        if normalize is not None:
            temp = [x * normalize / max(cluster_result) for x in cluster_result]
            cluster_result = temp
        result_stack.append(cluster_result)

    plt.clf()
    plt.title(file_name + f" ({number_of_clusters} clusters)")
    seaborn.heatmap(result_stack)
    plt.savefig(result_path)


def cluster_analysis(input_path, output_path, image_path):

    file_name, image_name = image_path.parts[-2], image_path.parts[-1]
    image_id = file_name + '\n' + image_name
    csv_id = file_name + ' - ' + image_name

    result_path = output_path / image_path.with_suffix('.png')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = result_path.with_suffix('.csv')

    image_stack = io.imread(input_path / image_path)
    xxx(image_stack, image_id, result_path, csv_id, csv_path)


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    run_processing(input_path, output_path, ['*.tif', '*.tiff'], cluster_analysis)


if __name__ == '__main__':
    cli()
