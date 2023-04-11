from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from readlif.reader import LifFile


def convert_lif_file(input_path: str | Path,
                     output_path: str | Path
                     ):

    input_path = Path(input_path)
    output_path = Path(output_path)

    lif_file_paths = [
        p.relative_to(input_path)
        for p in input_path.rglob('*.lif')
        if p.relative_to(input_path).parts[0] != '.git'
    ]

    for lif_file_path in lif_file_paths:
        write_images(
            input_path / lif_file_path,
            output_path / lif_file_path)


def write_images(input_path: Path, output_path: Path):

    lif_file = LifFile(input_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(lif_file.get_iter_image()):
        for suffix in ('tif', 'gif'):
            image_path = output_path / f'{image.name}.{suffix}'
            frame_list = [frame for frame in image.get_iter_t(c=0, z=0)]
            frame_list[0].save(
                fp=image_path,
                save_all=True,
                append_images=frame_list[1:],
                optimization=False)
            print(f'saved: {image_path}')


if __name__ == "__main__":

    convert_lif_file(
        Path('/home/cristian/datalad/neuron_data_1'),
        Path('/home/cristian/tmp/output'))
