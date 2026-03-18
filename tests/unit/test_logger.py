"""Unit tests for the logger module.

Adapted from stable-baselines3 tests/test_logger.py.
"""

from io import TextIOBase
import os
from pathlib import Path
import sys

import numpy as np
from pandas.errors import EmptyDataError
import pytest
import torch as th

from lit_sb3.utils.logger import (
    CSVOutputFormat,
    Figure,
    FormatUnsupportedError,
    HParam,
    HumanOutputFormat,
    Image,
    Logger,
    TensorBoardOutputFormat,
    Video,
    configure,
    make_output_format,
    read_csv,
    read_json,
)

KEY_VALUES = {
    "test": 1,
    "b": -3.14,
    "8": 9.9,
    "l": [1, 2],
    "a": np.array([1, 2, 3]),
    "f": np.array(1),
    "g": np.array([[[1]]]),
    "h": 'this ", ;is a \n tes:,t',
    "i": th.ones(3),
}

KEY_EXCLUDED = dict.fromkeys(KEY_VALUES.keys())


class LogContent:
    """Simple wrapper to check content for emptiness."""

    def __init__(self, _format: str, lines):
        self.format = _format
        self.lines = lines

    @property
    def empty(self):
        return len(self.lines) == 0


@pytest.fixture
def read_log(tmp_path, capsys):
    def read_fn(_format):
        if _format == "csv":
            try:
                df = read_csv(tmp_path / "progress.csv")
            except EmptyDataError:
                return LogContent(_format, [])
            return LogContent(_format, [r for _, r in df.iterrows() if not r.empty])
        elif _format == "json":
            try:
                df = read_json(tmp_path / "progress.json")
            except EmptyDataError:
                return LogContent(_format, [])
            return LogContent(_format, [r for _, r in df.iterrows() if not r.empty])
        elif _format == "stdout":
            captured = capsys.readouterr()
            return LogContent(_format, captured.out.splitlines())
        elif _format == "log":
            return LogContent(_format, (tmp_path / "log.txt").read_text().splitlines())
        elif _format == "tensorboard":
            from tensorboard.backend.event_processing.event_accumulator import (  # ty:ignore[unresolved-import]
                EventAccumulator,
            )

            acc = EventAccumulator(str(tmp_path))
            acc.Reload()
            tb_values_logged = []
            for reservoir in [acc.scalars, acc.tensors, acc.images, acc.histograms, acc.compressed_histograms]:
                for k in reservoir.Keys():
                    tb_values_logged.append(f"{k}: {reservoir.Items(k)!s}")
            return LogContent(_format, tb_values_logged)

    return read_fn


def test_main(tmp_path):
    """Tests for the logger module."""
    from lit_sb3.utils.logger import DEBUG, INFO

    logger = configure(None, ["stdout"])
    logger.info("hi")
    logger.debug("shouldn't appear")
    assert logger.level == INFO

    logger.set_level(DEBUG)
    assert logger.level == DEBUG
    logger.debug("should appear")

    logger = configure(folder=str(tmp_path))
    assert logger.dir == str(tmp_path)
    logger.record("a", 3)
    logger.record("b", 2.5)
    logger.dump()
    logger.record("b", -2.5)
    logger.record("a", 5.5)
    logger.dump()
    logger.info("^^^ should see a = 5.5")
    logger.record("f", "this text \n \r should appear in one line")
    logger.dump()
    logger.record_mean("b", -22.5)
    logger.record_mean("b", -44.4)
    logger.record("a", 5.5)
    logger.record("hist1", th.ones(2))
    logger.record("hist2", np.ones(2))
    logger.dump()
    logger.record("a", "longasslongasslongasslongasslongasslongassvalue")
    logger.dump()
    logger.warn("hey")
    logger.error("oh")


@pytest.mark.parametrize("_format", ["stdout", "log", "json", "csv", "tensorboard"])
def test_make_output(tmp_path, read_log, _format):
    """Test make_output_format for each format."""
    if _format == "tensorboard":
        pytest.importorskip("tensorboard")

    writer = make_output_format(_format, tmp_path)
    writer.write(KEY_VALUES, KEY_EXCLUDED)
    assert not read_log(_format).empty
    writer.close()


def test_make_output_fail(tmp_path):
    """Test value error on unsupported logger format."""
    with pytest.raises(ValueError):
        make_output_format("dummy_format", tmp_path)


@pytest.mark.parametrize("_format", ["stdout", "log", "json", "csv", "tensorboard"])
@pytest.mark.filterwarnings("ignore:Tried to write empty key-value dict")
def test_exclude_keys(tmp_path, read_log, _format):
    if _format == "tensorboard":
        pytest.importorskip("tensorboard")

    writer = make_output_format(_format, tmp_path)
    writer.write({"some_tag": 42}, key_excluded={"some_tag": (_format)})
    writer.close()
    assert read_log(_format).empty


@pytest.mark.parametrize("unsupported_format", ["stdout", "log", "json", "csv"])
def test_unsupported_video_format(tmp_path, unsupported_format):
    writer = make_output_format(unsupported_format, tmp_path)
    with pytest.raises(FormatUnsupportedError) as exec_info:
        video = Video(frames=th.rand(1, 20, 3, 16, 16), fps=20)
        writer.write({"video": video}, key_excluded={"video": ()})
    assert unsupported_format in str(exec_info.value)
    writer.close()


@pytest.mark.parametrize("unsupported_format", ["stdout", "log", "json", "csv"])
def test_unsupported_image_format(tmp_path, unsupported_format):
    writer = make_output_format(unsupported_format, tmp_path)
    with pytest.raises(FormatUnsupportedError) as exec_info:
        image = Image(image=th.rand(16, 16, 3), dataformats="HWC")
        writer.write({"image": image}, key_excluded={"image": ()})
    assert unsupported_format in str(exec_info.value)
    writer.close()


@pytest.mark.parametrize("unsupported_format", ["stdout", "log", "json", "csv"])
def test_unsupported_figure_format(tmp_path, unsupported_format):
    from matplotlib import pyplot as plt

    writer = make_output_format(unsupported_format, tmp_path)
    with pytest.raises(FormatUnsupportedError) as exec_info:
        fig = plt.figure()
        fig.add_subplot().plot(np.random.random(3))
        figure = Figure(figure=fig, close=True)
        writer.write({"figure": figure}, key_excluded={"figure": ()})
    assert unsupported_format in str(exec_info.value)
    writer.close()


@pytest.mark.parametrize("unsupported_format", ["stdout", "log", "json", "csv"])
def test_unsupported_hparam(tmp_path, unsupported_format):
    writer = make_output_format(unsupported_format, tmp_path)
    with pytest.raises(FormatUnsupportedError) as exec_info:
        hparam_dict = {"learning rate": np.random.random()}
        metric_dict = {"train/value_loss": 0}
        hparam = HParam(hparam_dict=hparam_dict, metric_dict=metric_dict)
        writer.write({"hparam": hparam}, key_excluded={"hparam": ()})
    assert unsupported_format in str(exec_info.value)
    writer.close()


def test_key_length(tmp_path):
    writer = make_output_format("stdout", tmp_path)
    assert writer.max_length == 36  # ty:ignore[unresolved-attribute]
    long_prefix = "a" * writer.max_length  # ty:ignore[unresolved-attribute]

    ok_dict = {
        "a" + long_prefix: 42,
        "b" + long_prefix: 42,
        "foobar": long_prefix + "a",
        "fizzbuzz": long_prefix + "b",
    }
    ok_excluded = dict.fromkeys(ok_dict)
    writer.write(ok_dict, ok_excluded)

    long_key_dict = {
        long_prefix + "a": 42,
        "foobar": "sdf",
        long_prefix + "b": 42,
    }
    long_key_excluded = dict.fromkeys(long_key_dict)
    with pytest.raises(ValueError, match=r"Key.*truncated"):
        writer.write(long_key_dict, long_key_excluded)

    writer.max_length += 1  # ty:ignore[unresolved-attribute]
    writer.write(long_key_dict, long_key_excluded)


@pytest.mark.parametrize("base_class", [object, TextIOBase])
def test_human_out_custom_text_io(base_class):
    class DummyTextIO(base_class):
        def __init__(self) -> None:
            super().__init__()
            self.lines = [[]]

        def write(self, t: str) -> int:
            self.lines[-1].append(t)
            return len(t)

        def flush(self) -> None:
            self.lines.append([])

        def close(self) -> None:
            pass

        def get_printed(self) -> str:
            return "\n".join(["".join(line) for line in self.lines])

    dummy_text_io = DummyTextIO()
    output = HumanOutputFormat(dummy_text_io)
    output.write({"key1": "value1", "key2": 42}, {"key1": None, "key2": None})
    output.write({"key1": "value2", "key2": 43}, {"key1": None, "key2": None})
    printed = dummy_text_io.get_printed()
    desired_printed = """-----------------
| key1 | value1 |
| key2 | 42     |
-----------------

-----------------
| key1 | value2 |
| key2 | 43     |
-----------------

"""
    assert printed == desired_printed


def test_human_output_same_keys_different_tags():
    human_out = HumanOutputFormat(sys.stdout, max_length=60)
    human_out.write(
        {"key1/foo": "value1", "key1/bar": "value2", "key2/bizz": "value3", "key2/foo": "value4"},
        {"key1/foo": None, "key2/bizz": None, "key1/bar": None, "key2/foo": None},
    )


def test_configure_creates_dir(tmp_path):
    """Test that configure() creates the target directory."""
    log_dir = str(tmp_path / "new_subdir")
    logger = configure(folder=log_dir)
    assert logger.dir == log_dir
    assert Path(log_dir).is_dir()
