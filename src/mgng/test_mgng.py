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
from __future__ import print_function, division

import unittest

import mgng2 as mgng
import networkx as nx
import numpy as np


class TestMGNG(unittest.TestCase):

    def setUp(self):
        self.inst = mgng.MGNG(dimensions=1, alpha=0.13, beta=0.7, delta=0.7,
                              gamma=5, theta=100, eta=0.7, lmbda=5, e_w=1,
                              e_n=1)

    def tearDown(self):
        del self.inst

    '''
    expected result: 14.44 = (1 - 0.13) * ||7 - 3||^2 + 0.13*||3 - 5||^2
    '''
    def test_distances(self):
        xt = 7
        self.inst.c_t = 3
        self.inst.weights = np.array([[3], [np.nan]])
        self.inst.contexts = np.array([[5], [np.nan]])
        actual = self.inst.distances(xt)[0]
        self.assertEquals(14.44, actual)

    '''
    expected result: node added to model
    '''
    def test_add_node(self):
        actual = self.inst._add_node(e=1, w=2, c=3)
        self.assertEquals(2, actual)
        self.assertEquals(1, self.inst.errors[actual])
        self.assertEquals(2, self.inst.weights[actual])
        self.assertEquals(3, self.inst.contexts[actual])

    '''
    expected result: 1st Neuron.i = 0, 2nd Neuron.i = 1
    '''
    def test_find_winners(self):
        xt = 7
        expected = [self.inst._add_node(e=0, w=3, c=5),
                    self.inst._add_node(e=0, w=1, c=3)]
        self.inst._add_node(e=0, w=23, c=29)
        self.inst._add_node(e=0, w=17, c=19)
        self.inst.c_t = 3

        actual = self.inst.find_winner_neurons(xt)
        self.assertEquals(expected[0], actual[0][1])
        self.assertEquals(expected[1], actual[1][1])

    '''
    expected result:
        r[w] = 3 + 0.7 * (7 - 3) = 5.8
        r[c] = 5 + 0.7 * (11 - 5) = 9.2
        N1[w] = 23 + 0.05 * (7 - 23) = 22.2
        N1[c] = 29 + 0.05 * (11 - 29) = 28.1
        N2[w] = 17 + 0.05 * (7 - 17) = 16.5
        N2[c] = 19 + 0.05 * (11 - 19) = 18.6
    '''
    def test_update_neighbors(self):
        xt = 7
        r = self.inst._add_node(e=0, w=3, c=5)
        u = self.inst._add_node(e=0, w=23, c=29)
        v = self.inst._add_node(e=0, w=17, c=19)
        self.inst._add_edge(r, u)
        self.inst._add_edge(r, v)
        self.inst.c_t = 11
        self.inst.e_w = 0.7
        self.inst.e_n = 0.05

        self.inst._update_neighbors(r, xt)
        self.assertEquals(5.8, self.inst.weights[r])
        self.assertEquals(9.2, self.inst.contexts[r])
        self.assertEquals([u, v], self.inst.model.neighbors(r))
        self.assertEquals(22.2, self.inst.weights[u])
        self.assertEquals(28.1, self.inst.contexts[u])
        self.assertEquals(16.5, self.inst.weights[v])
        self.assertEquals(18.6, self.inst.contexts[v])

    '''
    expected resut: 1st edge.age = 3; 2nd edge.age = 4
    '''
    def test_increment_edges_age(self):
        r = self.inst._add_node(e=0, w=3, c=5)
        u = self.inst._add_node(e=0, w=23, c=29)
        v = self.inst._add_node(e=0, w=17, c=19)
        self.inst.model.add_edge(r, u, age=2)
        self.inst.model.add_edge(r, v, age=3)

        self.inst._increment_edges_age(r)
        self.assertEquals(3, self.inst.model[r][u]['age'])
        self.assertEquals(4, self.inst.model[r][v]['age'])

    '''
    expected resut: 1 edge added between r and s, with age=0
    '''
    def test_add_edge(self):
        r = self.inst._add_node(e=0, w=3, c=5)
        s = self.inst._add_node(e=0, w=23, c=29)

        self.inst._add_edge(r, s)
        self.assertEquals((r, s, {'age': 0}),
                          self.inst.model.edges(data=True)[0])

    '''
    expected result: edge 0 not deleted, edge 1 present
    '''
    def test_remove_old_edges(self):
        r = self.inst._add_node(e=0, w=3, c=5)
        s = self.inst._add_node(e=0, w=3, c=5)
        t = self.inst._add_node(e=0, w=3, c=5)
        u = self.inst._add_node(e=0, w=3, c=5)
        self.inst._add_edge(r, s)
        self.inst._add_edge(t, u)
        self.inst.model[r][s]['age'] = 6

        self.inst._remove_old_edges()
        self.assertNotIn((r, s), self.inst.model.edges())
        self.assertIn((t, u), self.inst.model.edges())

    '''
    '''
    def test_remove_unconnected_neurons(self):
        r = self.inst._add_node(e=0, w=3, c=5)
        s = self.inst._add_node(e=0, w=3, c=5)
        t = self.inst._add_node(e=0, w=3, c=5)
        u = self.inst._add_node(e=0, w=3, c=5)
        self.inst._add_edge(t, u)

        self.inst._remove_unconnected_neurons()
        self.assertNotIn(r, self.inst.model.nodes())
        self.assertNotIn(s, self.inst.model.nodes())
        self.assertNotIn((r, s), self.inst.model.edges())
        self.assertIn(t, self.inst.model.nodes())
        self.assertIn(u, self.inst.model.nodes())
        self.assertIn((t, u), self.inst.model.edges())

    '''
    expected result:
        create new node if t mod \lambda = 0 and |K| < \theta
        new neuron created between neuron q with greatest e_q and
        q's neibouring neuron f with greatest e_f
        K := K \cup l
        w_l := 1/2 * (w_q + w_f)
        c_l := 1/2 * (c_q + c_f)
        e_l := \delta * (e_f + e_q)
        e_q := (1 - \deta) * e_q
        e_f := (1 - \deta) * e_f
    '''
    def test_create_new_neuron(self):
        q = self.inst._add_node(e=5, w=3, c=5)
        f = self.inst._add_node(e=3, w=3, c=5)
        s = self.inst._add_node(e=2, w=3, c=5)
        t = self.inst._add_node(e=4, w=3, c=5)
        u = self.inst._add_node(e=1, w=3, c=5)
        self.inst._add_edge(q, s)
        self.inst._add_edge(q, f)
        self.inst._add_edge(t, u)

        expt_w_l = (self.inst.weights[q] + self.inst.weights[f]) / 2.
        expt_c_l = (self.inst.contexts[q] + self.inst.contexts[f]) / 2.
        expt_e_l = self.inst.delta * (self.inst.errors[f] + self.inst.errors[q])
        expt_e_q = (1 - self.inst.delta) * self.inst.errors[q]
        expt_e_f = (1 - self.inst.delta) * self.inst.errors[f]
        l = self.inst._create_new_neuron()

        self.assertIn(l, self.inst.model.node)
        self.assertEquals(2, len(self.inst.model.neighbors(l)))
        self.assertIn(q, self.inst.model.neighbors(l))
        self.assertIn(f, self.inst.model.neighbors(l))
        self.assertEquals(expt_w_l, self.inst.weights[l])
        self.assertEquals(expt_c_l, self.inst.contexts[l])
        self.assertEquals(expt_e_l, self.inst.errors[l])
        self.assertEquals(expt_e_q, self.inst.errors[q])
        self.assertEquals(expt_e_f, self.inst.errors[f])

    '''
    expected result:
        * new edge created between r and s
        * number of neurons = 3
    '''
    def test_time_step(self):
        q = self.inst._add_node(e=5, w=23, c=25)
        r = self.inst._add_node(e=5, w=3, c=5)
        s = self.inst._add_node(e=3, w=5, c=7)
        self.inst.c_t = 3

        self.inst.t = 5
        self.inst.time_step(2)
        self.assertEquals(3, len(self.inst.model.nodes()))
        self.assertEquals(2, len(self.inst.model.edges()))

if __name__ == '__main__':
    unittest.main()
