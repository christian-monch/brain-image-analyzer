from __future__ import annotations

import dataclasses
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from seaborn import heatmap

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

argument_parser.add_argument(
    '-t',
    '--tiles',
    type=int,
    default=8,
    dest='tiles',
    help='how rows and columns of tiles (default: 8)')


cmap = plt.cm.inferno
alpha_cmap = cmap(np.arange(cmap.N))
alpha_cmap[:, -1] = np.linspace(0, 1, cmap.N)
alpha_cmap = ListedColormap(alpha_cmap)


pulse_parameter = PulseParameter()
tile_parameter = TileParameter(number_of_tiles=8)


def get_diff_sets(set_list: list[set]) -> set[frozenset]:
    """ determine how many different there are """

    if len(set_list) == 0:
        return set()

    if len(set_list) == 1:
        return {frozenset(set_list[0])}

    differing_sets = set_list[:]
    restart = True
    while restart is True:
        restart = False
        for index_a in range(len(differing_sets) - 1):
            pulse_set_a = differing_sets[index_a]
            for index_b in range(index_a + 1, len(differing_sets)):
                pulse_set_b = differing_sets[index_b]

                if pulse_set_a == pulse_set_b:
                    continue

                # We consider contained sets to be equivalent to the containing set
                x = pulse_set_a.union(pulse_set_b)
                if x == pulse_set_a or x == pulse_set_b:
                    differing_sets[index_a] = x
                    differing_sets[index_b] = x
                    restart = True

                # Everything else is considered to be different

            if restart is True:
                break

    result = set([frozenset(s) for s in differing_sets])
    return result


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
    pulse_sets = list()

    for frame in range(frames):
        all_tile_pulses = [
            (tiles * int(r / y_per_tile) + int(c / x_per_tile), pulse_info)
            for (f, r, c), pulse_info in all_pulses.items()
            if f == frame]

        if not all_tile_pulses:
            continue

        # Add the pulse set (should be something like set(select(0, ...))
        pulse_sets.append({pulse[0] for pulse in all_tile_pulses})

        for tile in range(tiles * tiles):
            tile_pulses = [
                pulse_info
                for t, pulse_info in all_tile_pulses
                if t == tile]
            if not tile_pulses:
                continue
            tile_map[tile][frame] = len(tile_pulses)

    tiles_max_pulses = np.max(tile_map)
    if tiles_max_pulses != 0:
        tile_map = 100 / tiles_max_pulses * tile_map
        different_sets = get_diff_sets(pulse_sets)
    else:
        tile_map = np.zeros([tiles * tiles, frames])
        different_sets = []

    # Write an image ot file
    plt.clf()
    plt.title(file_name + f" ({len(different_sets)} shapes)")
    heatmap(tile_map, cmap='hot', annot=False)
    plt.savefig(result_path_stem.with_suffix('.png'))

    # Write csv to file


def cli():

    arguments = argument_parser.parse_args()
    input_path = Path(arguments.image_path)
    output_path = Path(arguments.output_path)

    tile_parameter.number_of_tiles = arguments.tiles

    run_processing(
        input_path,
        output_path,
        ['*.tif', '*.tiff'],
        {p.get_name(): dataclasses.asdict(p) for p in [pulse_parameter, tile_parameter]},
        tiles_analysis
    )


if __name__ == '__main__':
    cli()
