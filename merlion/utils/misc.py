#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from abc import ABCMeta
from collections import OrderedDict
import importlib

import inspect
from functools import wraps


class AutodocABCMeta(ABCMeta):
    """
    Metaclass used to ensure that inherited members of an abstract base class
    also inherit docstrings for inherited methods.
    """

    def __new__(mcls, classname, bases, cls_dict):
        cls = super().__new__(mcls, classname, bases, cls_dict)
        for name, member in cls_dict.items():
            if member.__doc__ is None:
                for base in bases[::-1]:
                    attr = getattr(base, name, None)
                    if attr is not None:
                        member.__doc__ = attr.__doc__
                        break
        return cls


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
