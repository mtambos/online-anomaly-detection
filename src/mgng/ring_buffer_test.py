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
import sys
import unittest

import numpy as np

import ring_buffer


class RingBufferTests(unittest.TestCase):
    def test_extend_less_than_len_1d(self):
        #unidimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros(10))
        expected = np.random.random(8)
        ringbuff.extend(expected)
        self.assertEqual(expected.tolist(),
                         ringbuff[-len(expected):].tolist())

    def test_extend_less_than_len_nd(self):
        #n-dimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros((10, 4)))
        expected = np.random.random((8, 4))
        ringbuff.extend(expected)
        self.assertEqual(expected.tolist(),
                         ringbuff[-len(expected):].tolist())

    def test_extend_equal_len_1d(self):
        #unidimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros(10))
        expected = np.random.random(10)
        ringbuff.extend(expected)
        self.assertEqual(expected.tolist(), ringbuff.tolist())

    def test_extend_equal_len_nd(self):
        #n-dimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros((10, 4)))
        expected = np.random.random((10, 4))
        ringbuff.extend(expected)
        self.assertEqual(expected.tolist(), ringbuff.tolist())

    def test_extend_more_than_len_1d(self):
        #unidimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros(10))
        expected = np.random.random(12)
        ringbuff.extend(expected)
        self.assertEqual(expected[-len(ringbuff):].tolist(),
                         ringbuff.tolist())

    def test_extend_more_than_len_nd(self):
        #n-dimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros((10, 4)))
        expected = np.random.random((12, 4))
        ringbuff.extend(expected)
        self.assertEqual(expected[-len(ringbuff):].tolist(),
                         ringbuff.tolist())

    def test_append_less_than_len_1d(self):
        #unidimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros(10))
        expected = np.random.random(8)
        for i in range(len(expected)):
            ringbuff.append(expected[i])
        self.assertEqual(expected.tolist(),
                         ringbuff[-len(expected):].tolist())

    def test_append_less_than_len_nd(self):
        #n-dimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros((10, 4)))
        expected = np.random.random((8, 4))
        for i in range(len(expected)):
            ringbuff.append(expected[i])
        self.assertEqual(expected.tolist(),
                         ringbuff[-len(expected):].tolist())

    def test_extend_equal_len_1d(self):
        #unidimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros(10))
        expected = np.random.random(10)
        for i in range(len(expected)):
            ringbuff.append(expected[i])
        self.assertEqual(expected.tolist(), ringbuff.tolist())

    def test_extend_equal_len_nd(self):
        #n-dimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros((10, 4)))
        expected = np.random.random((10, 4))
        for i in range(len(expected)):
            ringbuff.append(expected[i])
        self.assertEqual(expected.tolist(), ringbuff.tolist())

    def test_extend_more_than_len_1d(self):
        #unidimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros(10))
        expected = np.random.random(12)
        for i in range(len(expected)):
            ringbuff.append(expected[i])
        self.assertEqual(expected[-len(ringbuff):].tolist(),
                         ringbuff.tolist())

    def test_extend_more_than_len_nd(self):
        #n-dimensional case
        ringbuff = ring_buffer.RingBuffer(np.zeros((10, 4)))
        expected = np.random.random((12, 4))
        for i in range(len(expected)):
            ringbuff.append(expected[i])
        self.assertEqual(expected[-len(ringbuff):].tolist(),
                         ringbuff.tolist())

if __name__ == '__main__':
    unittest.main()
