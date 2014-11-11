#!/usr/bin/env python
# ----------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------
'''
@author: Mario Tambos
'''
import numpy as np


class RingBuffer(np.ndarray):
    'A multidimensional ring buffer.'
    
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def extend(self, xs):
        'Adds array xs to the ring buffer. If xs is longer than the ring '
        'buffer, the last len(ring buffer) of xs are added the ring buffer.'
        xs = np.asarray(xs)
        if self.shape[1:] != xs.shape[1:]:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "xs.shape={}".format(self.shape, xs.shape))
        len_self = len(self)
        len_xs = len(xs)
        if len_self <= len_xs:
            xs = xs[-len_self:]
            len_xs = len(xs)
        else:
            self[:-len_xs] = self[len_xs:]
        self[-len_xs:] = xs

    def append(self, x):
        'Adds element x to the ring buffer.'
        x = np.asarray(x)
        if self.shape[1:] != x.shape:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "x.shape={}, x=".format(self.shape, x.shape, x))
        len_self = len(self)
        self[:-1] = self[1:]
        self[-1] = x
