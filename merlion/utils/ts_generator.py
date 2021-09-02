#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd

from operator import add
from functools import reduce
from typing import Callable, List, Union

from merlion.utils.time_series import UnivariateTimeSeries, TimeSeries


class TimeSeriesGenerator:
    """
    An abstract base class for generating synthetic time series data.
    """

    def __init__(
        self,
        f: Callable[[float], float],
        n: int,
        x0: float = 0.0,
        step: float = 1.0,
        scale: float = 1.0,
        noise: Callable[[], float] = np.random.normal,
        distort: Callable[[float, float], float] = add,
        name: str = None,
        t0: str = "1970 00:00:00",
        tdelta: str = "5min",
    ):
        """
        :param n: The number of points to be generated.
        :param x0: The initial value to use to form that 1-dimensional grid that
            will be used to compute the synthetic values.
        :param step: The step size to use when forming the 1-dimensional grid.
        :param scale: A scalar to use to either inflate or deflate the synthetic data.
        :param noise: A function that generates a random value when called.
        :param distort: A function mapping two real numbers to one real number which will
            be used to inject noise into the time series.
        :param name: The name to assign the univariate that will be generated.
        :param t0: Initial timestamp to use when wrapping the generated values into a
            TimeSeries object.
        :param tdelta: the time delta to use when wrapping the generated values into a
            TimeSeries object.

        Generates a 1-dimensional grid x(0), x(1), ..., x(n-1), where x(i) = x0 + i * step.
        Then generates a time series y(0), y(1), ..., y(n-1), where y(i) = f(x(i)) + noise.
        """
        assert step > 0, f"step must be a postive real number but is {step}."
        assert scale > 0, f"scale must be a postive real number but is {scale}."
        self.f = f
        self.n = n
        self.x0 = x0
        self.step = step
        self.scale = scale
        self.noise = noise
        self.distort = distort
        self.name = name
        self.t0 = t0
        self.tdelta = tdelta

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n: int):
        self._n = n
        self._update_steps()

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x: float):
        self._x0 = x
        self._update_steps()

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step: float):
        self._step = step
        self._update_steps()

    def _update_steps(self):
        """
        Updates the x-steps that are used to generate the time series
        based on the current values of `n`, `x0`, and `step`.
        """
        if all(hasattr(self, attr) for attr in ("_n", "_x0", "_step")):
            self.steps = [self.x0 + self.step * x for x in range(self.n)]

    def y(self, x: float):
        return self.scale * self.distort(self.f(x), self.noise())

    def generate(self, return_ts=True) -> Union[List[float], TimeSeries]:
        """
        Generates synthetic time series data according and returns it as a list or as a
        TimeSeries object.
        """
        vals = self._generate()
        if return_ts:
            assert self.t0 is not None and self.tdelta is not None
            times = pd.date_range(self.t0, periods=self.n, freq=self.tdelta)
            return UnivariateTimeSeries(times, vals, self.name).to_ts()

        return vals

    def _generate(self):
        return [self.y(x) for x in self.steps]


class GeneratorComposer(TimeSeriesGenerator):
    """
    A class for generating synthetic time series by composing
    other TimeSeriesGenerator's.
    """

    def __init__(self, generators: List[TimeSeriesGenerator], per_generator_noise: bool = False, **kwargs):
        kwargs["f"] = lambda x: x
        super().__init__(**kwargs)
        """
        :param generators: A list of other time series generators to compose.
        :param per_generator_noise: True if noise should be injected by each generator
            during composition. i.e., if we have two generators with generating functions
            f and g. If per_generator_noise = True, y = f(g+noise)+noise. Otherwise, 
            y = f(g) + noise.
        """
        self.per_generator_noise = per_generator_noise
        self.generators = generators

    @property
    def generators(self):
        return self._generators

    @generators.setter
    def generators(self, generators: List[TimeSeriesGenerator]):
        """
        Sets the generators for the GeneratorSequence.

        :param generators: The list of generators to set. Note that generators'
            attributes related to forming `steps` will not be relevant.
        """
        if self.per_generator_noise:
            self.noise = lambda: 0
        else:
            for generator in generators:
                generator.noise = lambda: 0
        self.f = reduce(lambda f, g: lambda x: f(g(x)), [g.f for g in generators], lambda x: x)


class GeneratorConcatenator(GeneratorComposer):
    """
    A class for generating synthetic time series data that undergoes
    fundamental changes to it's behavior that certain points in time.
    For example, with this class one could generate a time series that begins
    as linear and then becomes stationary.
    """

    def __init__(self, string_outputs: bool = True, **kwargs):
        """
        param string_outputs: If True, ensure that the end and beginning of each
            pair of consecutive time series are connected. For example, Let there be
            two generating functions f, and g belonging to consecutive generators. If
            True, adjust g by a constant c such that f(x) = g(x) at the last point x
            that f uses to generate its series.

        For example, let f = 0 with for 3 steps 0,1,2 and g = 2 * x for the next three
        steps 3,4,5. generate() returns:
            [0, 0, 0, 6, 8, 10] if string_outputs is False
            [0, 0, 0, 2, 4, 6]  if string_outputs is True.

        """
        kwargs["f"] = None
        kwargs["n"] = 1
        self.string_outputs = string_outputs
        super().__init__(**kwargs)

    @GeneratorComposer.generators.setter
    def generators(self, generators: List[TimeSeriesGenerator]):
        """
        Sets the generators for the GeneratorSequence.

        :param generators: The list of generators to set. Note that the
            individual generators `step` and `x0` attributes will be overriden
            by the `step` and `x0` belonging to the GeneratorSequence.
        """
        for i, generator in enumerate(generators):
            if not self.per_generator_noise:
                generator.noise = self.noise
                generator.distort = self.distort
            if i == 0:
                generator.x0 = self.x0
                self.n = generator.n
            elif i > 0:
                generator.x0 = self.x0 + self.n * self.step
                self.n += generator.n
            generator.step = self.step
        self._generators = generators

    def y(self, x: float):
        """
        A Generator Sequence has no method `y`.
        """
        pass

    def _generate(self) -> Union[List[float], TimeSeries]:
        """
        Generates the time series by concatenating the time series
        generated by each generator
        """
        result = []
        for generator in self.generators:
            y = generator.generate(return_ts=False)
            if self.string_outputs and result:
                y = np.asarray(y) + result[-1] - generator.f(generator.x0 - self.step)
                y = y.tolist()
            result += y

        return result
