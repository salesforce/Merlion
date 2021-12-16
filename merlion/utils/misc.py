#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from abc import ABCMeta
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
import importlib
import inspect
import re
from typing import Union


class AutodocABCMeta(ABCMeta):
    """
    Metaclass used to ensure that inherited members of an abstract base class
    also inherit docstrings for inherited methods.
    """

    def __new__(mcs, classname, bases, cls_dict):
        cls = super().__new__(mcs, classname, bases, cls_dict)
        for name, member in cls_dict.items():
            if member.__doc__ is None:
                for base in bases[::-1]:
                    attr = getattr(base, name, None)
                    if attr is not None:
                        member.__doc__ = attr.__doc__
                        break
        return cls


class ModelConfigMeta(type):
    """
    Metaclass used to ensure that the function signatures for model `Config` initializers contain all
    relevant parameters, including those specified in the superclass. Also update docstrings accordingly.

    For example, the only parameter of the base class `Config` is ``transform``. `ForecasterConfig` adds the
    parameters ``max_forecast_steps`` and ``target_seq_index``. Because `Config` inherits from this metaclass,
    we can declare

    .. code::

        class ForecasterConfig(Config):

        def __init__(self, max_forecast_steps: int = None, target_seq_index: int = None, **kwargs):
            ...

    and have the function signature for `ForecasterConfig`'s initializer include the parameter ``transform``,
    even though we never declared it explicitly. Additionally, the docstring for ``transform`` is inherited
    from the base class.
    """

    def __new__(mcs, classname, bases, cls_dict):
        sig = None
        cls = super().__new__(mcs, classname, bases, cls_dict)
        prefix, suffix, params = None, None, OrderedDict()
        for cls_ in cls.__mro__:
            if isinstance(cls_, ModelConfigMeta):
                # Combine the __init__ signatures
                sig = combine_signatures(sig, inspect.signature(cls_.__init__))

                # Parse the __init__ docstring. Use the earliest prefix/param docstring in the MRO.
                prefix_, suffix_, params_ = parse_init_docstring(cls_.__init__.__doc__)
                if prefix is None and any([line != "" for line in prefix_]):
                    prefix = "\n".join(prefix_)
                if suffix is None and any([line != "" for line in suffix_]):
                    suffix = "\n".join(suffix_)
                for param, docstring_lines in params_.items():
                    if param not in params:
                        params[param] = "\n".join(docstring_lines).rstrip("\n")

        # Update the signature and docstring of __init__
        cls.__init__.__signature__ = sig
        params = OrderedDict((p, params[p]) for p in sig.parameters if p in params)
        cls.__init__.__doc__ = (prefix or "") + "\n" + "\n".join(params.values()) + "\n\n" + (suffix or "")
        return cls


def combine_signatures(sig1: Union[inspect.Signature, None], sig2: Union[inspect.Signature, None]):
    """
    Utility function which combines the signatures of two functions.
    """
    if sig1 is None:
        return sig2
    if sig2 is None:
        return sig1

    # Get all params from sig1
    sig1 = deepcopy(sig1)
    params = list(sig1.parameters.values())
    for n, param in enumerate(params):
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            break
    else:
        n = len(params)

    # Add non-overlapping params from sig2
    for param in sig2.parameters.values():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            break
        if param.name not in sig1.parameters:
            params.insert(n, param)
            n += 1

    return sig1.replace(parameters=params)


def parse_init_docstring(docstring):
    docstring_lines = [""] if docstring is None else docstring.split("\n")
    prefix, suffix, param_dict = [], [], OrderedDict()
    non_empty_lines = [line for line in docstring_lines if len(line) > 0]
    indent = 0 if len(non_empty_lines) == 0 else len(re.search(r"^\s*", non_empty_lines[0]).group(0))
    for line in docstring_lines:
        line = line[indent:]
        match = re.search(r":param\s*(\w+):", line)
        if match is not None:
            param = match.group(1)
            param_dict[param] = [line]
        elif len(param_dict) == 0:
            prefix.append(line)
        elif len(suffix) > 0 or re.match(r"^[^\s]", line):  # not starting a param doc, but un-indented --> suffix
            suffix.append(line)
        else:
            param_dict[list(param_dict.keys())[-1]].append(line)
    return prefix, suffix, param_dict


class ValIterOrderedDict(OrderedDict):
    """
    OrderedDict whose iterator goes over self.values() instead of self.keys().
    """

    def __iter__(self):
        return iter(self.values())


def dynamic_import(import_path: str, alias: dict = None):
    """
    Dynamically import a member from the specified module.

    :param import_path: syntax 'module_name:member_name',
        e.g. 'merlion.transform.normalize:PowerTransform'
    :param alias: dict which maps shortcuts for the registered classes, to their
        full import paths.
    :return: imported class
    """
    alias = dict() if alias is None else alias
    if import_path not in alias and ":" not in import_path:
        raise ValueError(
            "import_path should be one of {} or "
            'include ":", e.g. "merlion.transform.normalize:MeanVarNormalize" : '
            "got {}".format(set(alias), import_path)
        )
    if ":" not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)


def initializer(func):
    """
    Decorator for the __init__ method.
    Automatically assigns the parameters.
    """
    argspec = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(argspec.args[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)
        for name, default in zip(reversed(argspec.args), reversed(argspec.defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)
        func(self, *args, **kargs)

    return wrapper


class ProgressBar:
    def __init__(self, total: int, length: int = 40, decimals: int = 1, fill: str = "█"):
        """
        :param total: total iterations
        :param length: character length of bar
        :param decimals: positive number of decimals in percent complete
        :param fill: bar fill character
        """
        self.total = total
        self.length = length
        self.decimals = decimals
        self.fill = fill

    def print(self, iteration, prefix, suffix, end=""):
        """
        :param iteration: current iteration
        :param prefix: prefix string
        :param suffix: suffix string
        :param end: end character (e.g. ``"\\r"``, ``"\\r\\n"``)
        """
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        fill_len = self.length * iteration // self.total
        bar = self.fill * fill_len + "-" * (self.length - fill_len)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=end)
        if iteration >= self.total:
            print()
