from __future__ import annotations

import dataclasses
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN

from pulse_analyzer.driver import run_processing
from pulse_analyzer.tools import ClusterParameter, PulseParameter
from pulse_analyzer.tools.pulses import find_pulses, find_all_pulses


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


pulse_parameter = PulseParameter()
cluster_parameter = ClusterParameter()


def find_frame_cluster(frames,
                       all_pulses
                       ) -> list:
    """try to find cluster in individual frames"""
    cluster_per_frame = []
    for frame in range(frames):
        points = [(column, row) for (f, row, column) in all_pulses if f == frame]
        if not points:
            cluster_per_frame.append([])
            continue
        clustering = DBSCAN(eps=5).fit(points)
        labels = clustering.labels_
        clusters = [
            [(x, y) for label, (x, y) in zip(labels, points) if label == outer_label]
            for outer_label in set(labels)
            if outer_label != -1
        ]
        cluster_per_frame.append(clusters)
    return cluster_per_frame


def cluster_analysis(image_stack: np.ndarray,
                     file_name: str,
                     result_path_stem: Path,
                     show_images=True,
                     normalize: int | None = 100):

    frames, rows, columns = image_stack.shape
    all_pulses = find_all_pulses(image_stack, **(dataclasses.asdict(pulse_parameter)))
    cluster_per_frame = find_frame_cluster(frames, all_pulses)

    # DEBUG show all ffames and clusters
    for frame in range(frames):

        frame_cluster = cluster_per_frame[frame]
        if not frame_cluster:
            continue
        pulse_coord = [(c, r) for f, r, c in all_pulses.keys() if f == frame]

        plt.clf()
        plt.xlim(0, columns)
        plt.ylim(0, rows)
        plt.title(file_name + f'{frame}')

        for x, y in pulse_coord:
            plt.plot(x, y, 'r.')

        for cluster_index, cluster in enumerate(frame_cluster):
            for x, y in cluster:
                plt.plot(x, y, f'C{cluster_index % 10}.')

        plt.show()
        #plt.imshow(image_stack[0], vmin=0, vmax=255, cmap='gray', interpolation=None)
        #plt.imshow(all_pulses_image, vmin=0, vmax=255, alpha=0.9, cmap=alpha_cmap, interpolation='bilinear')

    return

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
    plt.savefig(result_path_stem.with_suffix('.png'))


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    run_processing(
        input_path,
        output_path,
        ['*.tif', '*.tiff'],
        {p.get_name(): dataclasses.asdict(p) for p in [pulse_parameter, cluster_parameter]},
        cluster_analysis
    )


if __name__ == '__main__':
    cli()
