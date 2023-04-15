from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import io

from pulse_analyzer.driver import run_processing
from pulse_analyzer.tools.pulses import find_pulses


argument_parser = ArgumentParser(
    prog='heatmap',
    description='Analyzes pulse events in image stacks. Output is a heatmap that shows how many events have occured at '
                'a pixel. The output is written as `.png`-image and as `.csv`-files.',
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


def get_heatmap_image(image_stack):
    rows, columns = image_stack[0].shape
    heatmap_image = np.zeros([rows, columns])
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            heatmap_image[r][c] = len(pulse_infos)
    return heatmap_image


def plot_heatmap_image(image_stack: ndarray,
                       image_id: str,
                       result_path: Path):

    heatmap_image = get_heatmap_image(image_stack)

    plt.clf()
    plt.title(image_id)
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.imshow(heatmap_image, cmap='hot', interpolation='nearest')
    plt.show()
    #plt.savefig(result_path)


def get_heatmap_image(image_stack):
    rows, columns = image_stack[0].shape
    heatmap_image = np.zeros([rows, columns])
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            heatmap_image[r][c] = len(pulse_infos)
    return heatmap_image


def plot_heatmap_images(input_path, output_path, image_path):

    file_name, image_name = image_path.parts[-2], image_path.parts[-1]
    image_id = file_name + '\n' + image_name

    result_path = output_path / image_path.with_suffix('.png')
    result_path.parent.mkdir(parents=True, exist_ok=True)

    image_stack = io.imread(input_path / image_path)
    plot_heatmap_image(image_stack, image_id, result_path)


def plot_heatmap_graph(image_stack: ndarray,
                       image_id: str,
                       result_path: Path):

    plt.clf()
    plt.title(image_id)
    plt.xlim(0, 512)
    plt.ylim(0, 512)

    max_pulses = 0
    rows, columns = image_stack[0].shape
    for r in range(rows):
        for c in range(columns):
            pulse_infos = find_pulses(image_stack, r, c)
            if pulse_infos:
                max_pulses = max([max_pulses, len(pulse_infos)])
                plt.plot(c, r, '.')

    plt.show()
    #plt.savefig(result_path)


def plot_heatmap_graphs(input_path, output_path, image_path):

    file_name, image_name = image_path.parts[-2], image_path.parts[-1]
    image_id = file_name + '\n' + image_name

    result_path = output_path / image_path.with_suffix('.png')
    result_path.parent.mkdir(parents=True, exist_ok=True)

    image_stack = io.imread(input_path / image_path)
    plot_heatmap_graph(image_stack, image_id, result_path)


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    run_processing(input_path, output_path, ['*.tif', '*.tiff'], plot_heatmap_graphs)


if __name__ == '__main__':
    cli()
