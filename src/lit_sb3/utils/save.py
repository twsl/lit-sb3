"""Save util taken from stable_baselines.

used to serialize data (class parameters) of model classes.
"""

import base64
import functools
import io
import json
import os
import pathlib
import pickle
from typing import Any, Optional, Union
import warnings
import zipfile

import cloudpickle
import torch

import lit_sb3
from lit_sb3.core.types import TensorDict
from lit_sb3.utils.device import get_device
from lit_sb3.utils.system import get_system_info


def recursive_getattr(obj: Any, attr: str, *args) -> Any:
    """Recursive version of getattr.

    taken from https://stackoverflow.com/questions/31174295.

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_getattr(MyObject, 'sub_object.name')  # return test

    Args:
        obj: Object to retrieve attribute from.
        attr: Attribute to retrieve.
        *args: Default value if attribute is not found.

    Returns:
        The attribute
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def recursive_setattr(obj: Any, attr: str, val: Any) -> None:
    """Recursive version of setattr.

    taken from https://stackoverflow.com/questions/31174295.

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_setattr(MyObject, 'sub_object.name', 'hello')

    Args:
        obj:
        attr: Attribute to set
        val: New value of the attribute
        obj: The obj parameter.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)


def is_json_serializable(item: Any) -> bool:
    """Test if an object is serializable into JSON.

    Args:
        item: The object to be tested for JSON serialization.

    Returns:
        True if object is JSON serializable, false otherwise.
    """
    json_serializable = True
    try:
        _ = json.dumps(item)
    except TypeError:
        json_serializable = False
    return json_serializable


def data_to_json(data: dict[str, Any]) -> str:
    """Turn data (class parameters) into a JSON string for storing.

    Args:
        data: Dictionary of class parameters to be
            stored. Items that are not JSON serializable will be
            pickled with Cloudpickle and stored as bytearray in
            the JSON file

    Returns:
        JSON string of the data serialized.
    """
    serializable_data = {}
    for data_key, data_item in data.items():
        if is_json_serializable(data_item):
            serializable_data[data_key] = data_item
        else:
            base64_encoded = base64.b64encode(cloudpickle.dumps(data_item)).decode()
            cloudpickle_serialization = {
                ":type:": str(type(data_item)),
                ":serialized:": base64_encoded,
            }
            if hasattr(data_item, "__dict__") or isinstance(data_item, dict):
                item_generator = data_item.items if isinstance(data_item, dict) else data_item.__dict__.items
                for variable_name, variable_item in item_generator():
                    if is_json_serializable(variable_item):
                        cloudpickle_serialization[variable_name] = variable_item
                    else:
                        cloudpickle_serialization[variable_name] = str(variable_item)

            serializable_data[data_key] = cloudpickle_serialization
    json_string = json.dumps(serializable_data, indent=4)
    return json_string


def json_to_data(json_string: str, custom_objects: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn JSON serialization of class-parameters back into dictionary.

    Args:
        json_string: JSON serialization of the class-parameters
            that should be loaded.
        custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.

    Returns:
        Loaded class parameters.
    """
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects argument must be a dict or None")

    json_dict = json.loads(json_string)
    return_data = {}
    for data_key, data_item in json_dict.items():
        if custom_objects is not None and data_key in custom_objects:
            return_data[data_key] = custom_objects[data_key]
        elif isinstance(data_item, dict) and ":serialized:" in data_item:
            serialization = data_item[":serialized:"]
            try:
                base64_object = base64.b64decode(serialization.encode())
                deserialized_object = cloudpickle.loads(base64_object)
            except (RuntimeError, TypeError, AttributeError) as e:
                warnings.warn(
                    f"Could not deserialize object {data_key}. "
                    "Consider using `custom_objects` argument to replace "
                    "this object.\n"
                    f"Exception: {e}",
                    stacklevel=2,
                )
            else:
                return_data[data_key] = deserialized_object
        else:
            return_data[data_key] = data_item
    return return_data


@functools.singledispatch
def open_path(
    path: str | pathlib.Path | io.BufferedIOBase, mode: str, verbose: int = 0, suffix: str | None = None
) -> io.BufferedWriter | io.BufferedReader | io.BytesIO | io.BufferedRandom:
    """Opens a path for reading or writing with a preferred suffix and raises debug information.

    If the provided path is a derivative of io.BufferedIOBase it ensures that the file
    matches the provided mode, i.e. If the mode is read ("r", "read") it checks that the path is readable.
    If the mode is write ("w", "write") it checks that the file is writable.

    If the provided path is a string or a pathlib.Path, it ensures that it exists. If the mode is "read"
    it checks that it exists, if it doesn't exist it attempts to read path.suffix if a suffix is provided.
    If the mode is "write" and the path does not exist, it creates all the parent folders. If the path
    points to a folder, it changes the path to path_2. If the path already exists and verbose >= 2,
    it raises a warning.

    Args:
        path: the path to open.
            if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
            path actually exists. If path is a io.BufferedIOBase the path exists.
        mode: how to open the file. "w"|"write" for writing, "r"|"read" for reading.
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
            If mode is "r" then we attempt to open the path. If an error is raised and the suffix
            is not None, we attempt to open the path with the suffix.
    """
    allowed_types = (io.BufferedWriter, io.BufferedReader, io.BytesIO, io.BufferedRandom)
    if not isinstance(path, allowed_types):
        raise TypeError(f"Path {path} parameter has invalid type: expected one of {allowed_types}.")
    if path.closed:
        raise ValueError(f"File stream {path} is closed.")
    mode = mode.lower()
    try:
        mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
    except KeyError as e:
        raise ValueError("Expected mode to be either 'w' or 'r'.") from e
    if ((mode == "w") and not path.writable()) or ((mode == "r") and not path.readable()):
        error_msg = "writable" if mode == "w" else "readable"
        raise ValueError(f"Expected a {error_msg} file.")
    return path


@open_path.register(str)  # ty:ignore[invalid-argument-type]
def open_path_str(path: str, mode: str, verbose: int = 0, suffix: str | None = None) -> io.BufferedIOBase:
    """Open a path given by a string. If writing to the path, the function ensures.

    that the path exists.

    Args:
        path: the path to open. If mode is "w" then it ensures that the path exists
            by creating the necessary folders and renaming path if it points to a folder.
        mode: how to open the file. "w" for writing, "r" for reading.
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
            If mode is "r" then we attempt to open the path. If an error is raised and the suffix
            is not None, we attempt to open the path with the suffix.
    """
    return open_path_pathlib(pathlib.Path(path), mode, verbose, suffix)


@open_path.register(pathlib.Path)  # ty:ignore[invalid-argument-type]
def open_path_pathlib(path: pathlib.Path, mode: str, verbose: int = 0, suffix: str | None = None) -> io.BufferedIOBase:
    """Open a path given by a string. If writing to the path, the function ensures.

    that the path exists.

    Args:
        path: the path to check. If mode is "w" then it
            ensures that the path exists by creating the necessary folders and
            renaming path if it points to a folder.
        mode: how to open the file. "w" for writing, "r" for reading.
        verbose: Verbosity level: 0 for no output, 2 for indicating if path without suffix is not found when mode is "r"
        suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
            If mode is "r" then we attempt to open the path. If an error is raised and the suffix
            is not None, we attempt to open the path with the suffix.
    """
    if mode not in ("w", "r"):
        raise ValueError("Expected mode to be either 'w' or 'r'.")

    if mode == "r":
        try:
            return open_path(path.open("rb"), mode, verbose, suffix)
        except FileNotFoundError:
            if suffix is not None and suffix != "":
                newpath = pathlib.Path(f"{path}.{suffix}")
                if verbose >= 2:
                    warnings.warn(f"Path '{path}' not found. Attempting {newpath}.", stacklevel=2)
                path, suffix = newpath, None
            else:
                raise
    else:
        try:
            if path.suffix == "" and suffix is not None and suffix != "":
                path = pathlib.Path(f"{path}.{suffix}")
            if path.exists() and path.is_file() and verbose >= 2:
                warnings.warn(f"Path '{path}' exists, will overwrite it.", stacklevel=2)
            return open_path(path.open("wb"), mode, verbose, suffix)
        except IsADirectoryError:
            warnings.warn(f"Path '{path}' is a folder. Will save instead to {path}_2", stacklevel=2)
            path = pathlib.Path(f"{path}_2")
        except FileNotFoundError:
            warnings.warn(f"Path '{path.parent}' does not exist. Will create it.", stacklevel=2)
            path.parent.mkdir(exist_ok=True, parents=True)

    return open_path_pathlib(path, mode, verbose, suffix)


def save_to_zip_file(
    save_path: str | pathlib.Path | io.BufferedIOBase,
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    pytorch_variables: dict[str, Any] | None = None,
    verbose: int = 0,
) -> None:
    """Save model data to a zip archive.

    Args:
        save_path: Where to store the model.
            if save_path is a str or pathlib.Path ensures that the path actually exists.
        data: Class parameters being stored (non-PyTorch variables)
        params: Model parameters being stored expected to contain an entry for every
            state_dict with its name and the state_dict.
        pytorch_variables: Other PyTorch variables expected to contain name and value of the variable.
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    file = open_path(save_path, "w", verbose=0, suffix="zip")
    if data is not None:
        serialized_data = data_to_json(data)

    with zipfile.ZipFile(file, mode="w") as archive:
        if data is not None:
            archive.writestr("data", serialized_data)
        if pytorch_variables is not None:
            with archive.open("pytorch_variables.pth", mode="w", force_zip64=True) as pytorch_variables_file:
                torch.save(pytorch_variables, pytorch_variables_file)
        if params is not None:
            for file_name, dict_ in params.items():
                with archive.open(file_name + ".pth", mode="w", force_zip64=True) as param_file:
                    torch.save(dict_, param_file)
        archive.writestr("_lit_sb3_version", lit_sb3.__version__)
        archive.writestr("system_info.txt", get_system_info(print_info=False)[1])

    if isinstance(save_path, (str, pathlib.Path)):
        file.close()


def save_to_pkl(path: str | pathlib.Path | io.BufferedIOBase, obj: Any, verbose: int = 0) -> None:
    """Save an object to path creating the necessary folders along the way.

    If the path exists and is a directory, it will raise a warning and rename the path.
    If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.

    Args:
        path: the path to open.
            if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
            path actually exists. If path is a io.BufferedIOBase the path exists.
        obj: The object to save.
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    file = open_path(path, "w", verbose=verbose, suffix="pkl")
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    if isinstance(path, (str, pathlib.Path)):
        file.close()


def load_from_pkl(path: str | pathlib.Path | io.BufferedIOBase, verbose: int = 0) -> Any:
    """Load an object from the path. If a suffix is provided in the path, it will use that suffix.

    If the path does not exist, it will attempt to load using the .pkl suffix.

    Args:
        path: the path to open.
            if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
            path actually exists. If path is a io.BufferedIOBase the path exists.
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    file = open_path(path, "r", verbose=verbose, suffix="pkl")
    obj = pickle.load(file)  # noqa: S301
    if isinstance(path, (str, pathlib.Path)):
        file.close()
    return obj


def load_from_zip_file(
    load_path: str | pathlib.Path | io.BufferedIOBase,
    load_data: bool = True,
    custom_objects: dict[str, Any] | None = None,
    device: torch.device | str = "auto",
    verbose: int = 0,
    print_system_info: bool = False,
) -> tuple[dict[str, Any] | None, TensorDict, TensorDict | None]:
    """Load model data from a .zip archive.

    Args:
        load_path: Where to load the model from
        load_data: Whether we should load and return data
            (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
        custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        device: Device on which the code should run.
        verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        print_system_info: Whether to print or not the system info
            about the saved model.

    Returns:
        Class parameters, model state_dicts (aka "params", dict of state_dict)
        and dict of pytorch variables
    """
    file = open_path(load_path, "r", verbose=verbose, suffix="zip")

    device = get_device(device=device)

    try:
        with zipfile.ZipFile(file) as archive:
            namelist = archive.namelist()
            data = None
            pytorch_variables = None
            params = {}

            if print_system_info:
                if "system_info.txt" in namelist:
                    print("== SAVED MODEL SYSTEM INFO ==")
                    print(archive.read("system_info.txt").decode())
                else:
                    warnings.warn(
                        "The model was saved with SB3 <= 1.2.0 and thus cannot print system information.",
                        UserWarning,
                        stacklevel=2,
                    )

            if "data" in namelist and load_data:
                json_data = archive.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects)

            pth_files = [file_name for file_name in namelist if pathlib.Path(file_name).suffix == ".pth"]
            for file_path in pth_files:
                with archive.open(file_path, mode="r") as param_file:
                    file_content = io.BytesIO()
                    file_content.write(param_file.read())
                    file_content.seek(0)
                    th_object = torch.load(file_content, map_location=device, weights_only=False)
                    if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
                        pytorch_variables = th_object
                    else:
                        params[pathlib.Path(file_path).stem] = th_object
    except zipfile.BadZipFile as e:
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file") from e
    finally:
        if isinstance(load_path, (str, pathlib.Path)):
            file.close()
    return data, params, pytorch_variables
