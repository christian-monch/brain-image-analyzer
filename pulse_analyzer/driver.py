from __future__ import annotations

import json
from itertools import chain
from pathlib import Path
from typing import Callable, Iterable

import progressbar
from skimage import io


# This input/output adaptor is specific to the
# microscopy dataset
def input_output_adaptor(input_path: Path,
                         output_path: Path,
                         image_path: Path,
                         processor: Callable,
                         processor_args: list | tuple,
                         processor_kwargs: dict):

    file_name, image_name = image_path.parts[-2], image_path.parts[-1]
    image_id = file_name + '\n' + image_name

    result_path_stem = output_path / image_path.with_suffix('')
    result_path_stem.parent.mkdir(parents=True, exist_ok=True)

    image_stack = io.imread(input_path / image_path)
    processor(*([image_stack, image_id, result_path_stem] + processor_args), **processor_kwargs)


def run_processing(input_path: Path,
                   output_path: Path,
                   patterns: str | list[str],
                   parameter: dict[str, dict],
                   processor: Callable,
                   processor_args: list | tuple | None = None,
                   processor_kwargs: dict | None = None):

    patterns = [patterns] if isinstance(patterns, str) else patterns
    processor_args = list(processor_args or list())
    processor_kwargs = processor_kwargs or dict()

    image_paths = [
        p.relative_to(input_path)
        for p in chain(*[input_path.rglob(pattern) for pattern in patterns])
        if p.relative_to(input_path).parts[0] != '.git'
    ]

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'parameter.json').write_text(json.dumps(parameter, indent=4))

    bar = progressbar.ProgressBar(max_value=len(image_paths))
    for index, image_path in enumerate(image_paths):
        input_output_adaptor(input_path, output_path, image_path, processor, processor_args, processor_kwargs)
        bar.update(index + 1)
