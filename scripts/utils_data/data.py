#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: data.py
# Date: Monday, January 17th 2022
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2022 Amelia Villegas-Morcillo
###


import copy
import torch


class Data(object):
    """A plain old python object modeling the attributes.
    Args:
        x : Feature matrix.
        y : Label vector.

    The data object can be extented by any other optional data.
    """
    def __init__(self, x=None, y=None, **kwargs):
        self.x = x
        self.y = y
        # Optional attributes
        for key, item in kwargs.items():
            self[key] = item

    def __getitem__(self, key):
        # Gets the data of the attribute
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        # Sets the attribute key to value
        setattr(self, key, value)

    @property
    def keys(self):
        # Returns all names of attributes
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        # Returns the number of all present attributes
        return len(self.keys)

    def __contains__(self, key):
        # Returns True if the attribute key is present in the data
        return key in self.keys

    def __iter__(self):
        # Iterates over all present attributes in the data, yielding their
        # attribute names and content
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        # Iterates over all attributes keys in the data, yielding
        # their attribute names and content
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def apply(self, func, *keys):
        # Applies the function func to all tensor attributes
        for key, item in self(*keys):
            if torch.is_tensor(item):
                self[key] = func(item)
        return self

    def to(self, device, *keys):
        # Performs tensor dtype and/or device conversion to all attributes
        return self.apply(lambda x: x.to(device), *keys)
