from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import progressbar
from readlif.reader import LifFile


progressbar.streams.wrap_stderr()


argument_parser = ArgumentParser(
    prog='lif_splitter',
    description='Split `.lif`-files into image stacks. The program reads all *.lif-files that are in the directory '
                '`LIF_PATH`, it extracts the image stacks and writes them to `OUTPUT_PATH` as `.tif`-files. The input '
                'files are not altered.',
    epilog='for more information see: https://github.com/christian-monch/brain-image-analyzer')

argument_parser.add_argument(
    'lif_path',
    help='directory that contains `*.lif` files, all lif-files in the directory tree will be interpreted and the '
         'image stacks that it contains will be written')

argument_parser.add_argument(
    'output_path',
    help='directory that will contain the extracted image stacks as `.tif`-files. Existing files will be overriden')


def convert_lif_file(input_path: str | Path,
                     output_path: str | Path):

    input_path = Path(input_path)
    output_path = Path(output_path)

    lif_file_paths = [
        p.relative_to(input_path)
        for p in input_path.rglob('*.lif')
        if p.relative_to(input_path).parts[0] != '.git']

    bar = progressbar.ProgressBar(max_value=len(lif_file_paths))
    for index, lif_file_path in enumerate(lif_file_paths):
        write_images(
            input_path / lif_file_path,
            output_path / lif_file_path)
        bar.update(index + 1)


def write_images(input_path: Path, output_path: Path):

    lif_file = LifFile(input_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(lif_file.get_iter_image()):
        for suffix in ('tif',):
            image_path = output_path / f'{image.name}.{suffix}'
            frame_list = [frame for frame in image.get_iter_t(c=0, z=0)]
            frame_list[0].save(
                fp=image_path,
                save_all=True,
                append_images=frame_list[1:],
                optimization=False)


def cli():
    arguments = argument_parser.parse_args()
    convert_lif_file(
        Path(arguments.lif_path),
        Path(arguments.output_path))


if __name__ == '__main__':
    if __name__ == '__main__':
        cli()
