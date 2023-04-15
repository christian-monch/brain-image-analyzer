from __future__ import annotations

import dataclasses
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from pulse_analyzer.driver import run_processing
from pulse_analyzer.tools import TileParameter, PulseParameter
from pulse_analyzer.tools.pulses import find_all_pulses


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
tile_parameter = TileParameter()


def tiles_analysis(image_stack: np.ndarray,
                   file_name: str,
                   result_path_stem: Path,
                   show_images=True,
                   normalize: int | None = 100):

    frames, rows, columns = image_stack.shape
    all_pulses = find_all_pulses(image_stack, **(dataclasses.asdict(pulse_parameter)))

    tiles = tile_parameter.number_of_tiles
    tile_map = np.zeros([tiles * tiles, frames])
    x_per_tile = int(columns / tiles)
    y_per_tile = int(rows / tiles)
    for frame in range(frames):
        all_tile_pulses = [
            (tiles * int(r / y_per_tile) + int(c / x_per_tile), pulse_info)
            for (f, r, c), pulse_info in all_pulses.items()
            if f == frame]

        if not all_tile_pulses:
            continue

        for tile in range(tiles * tiles):
            tile_pulses = [
                pulse_info
                for t, pulse_info in all_tile_pulses
                if t == tile]
            if not tile_pulses:
                continue
            tile_map[tile][frame] = len(tile_pulses)

    plt.clf()
    plt.title(file_name)
    plt.xlim(0, 60)
    plt.ylim(0, tiles * tiles)
    plt.imshow(tile_map, cmap='hot', interpolation='bilinear')
    plt.show()


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    run_processing(
        input_path,
        output_path,
        ['*.tif', '*.tiff'],
        {p.get_name(): dataclasses.asdict(p) for p in [pulse_parameter, tile_parameter]},
        tiles_analysis
    )


if __name__ == '__main__':
    cli()
