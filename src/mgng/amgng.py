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
from __future__ import division, print_function
from collections import deque

import bottleneck as bn
import inspect
from numbapro import autojit
import numexpr as ne
import numpy as np
import numpy.linalg as lnp
from scipy.stats import norm

from cdf_table import CDF_TABLE
import mgng
from ring_buffer import RingBuffer


class AMGNG:

    def __init__(self, comparison_function, buffer_len, dimensions,
                 prest_gamma, prest_lmbda, prest_theta,
                 pst_gamma, pst_lmbda, pst_theta,
                 prest_alpha=0.5, prest_beta=0.5, prest_delta=0.5,
                 prest_eta=0.9995, prest_e_w=0.05, prest_e_n=0.0006,
                 pst_alpha=0.5, pst_beta=0.75, pst_delta=0.5,
                 pst_eta=0.9995, pst_e_w=0.05, pst_e_n=0.0006,
                 ma_window_len=None, ma_recalc_delay=1, ddof=1):

        values = inspect.getargvalues(inspect.currentframe())[3]
        print('Init parameters: {}'.format(values))
        self.comparison_function = comparison_function
        self.buffer_len = buffer_len
        self.dimensions = dimensions
        self.present = mgng.MGNG(dimensions=dimensions,
                                 gamma=int(prest_gamma),
                                 lmbda=int(prest_lmbda),
                                 theta=int(prest_theta),
                                 alpha=float(prest_alpha),
                                 beta=float(prest_beta),
                                 delta=float(prest_delta),
                                 eta=float(prest_eta),
                                 e_w=float(prest_e_w),
                                 e_n=float(prest_e_n))
        self.past = mgng.MGNG(dimensions=dimensions,
                              gamma=int(pst_gamma),
                              lmbda=int(pst_lmbda),
                              theta=int(pst_theta),
                              alpha=float(pst_alpha),
                              beta=float(pst_beta),
                              delta=float(pst_delta),
                              eta=float(pst_eta),
                              e_w=float(pst_e_w),
                              e_n=float(pst_e_n))
        # self.buffer = deque(maxlen=self.buffer_len)
        self.buffer = RingBuffer([[np.nan]*dimensions]*buffer_len)
        if ma_window_len is None:
            # self.ma_window = deque(maxlen=self.buffer_len)
            self.ma_window = RingBuffer([np.nan]*buffer_len)
        else:
            # self.ma_window = deque(maxlen=ma_window_len)
            self.ma_window = RingBuffer([np.nan]*ma_window_len)
        self.ma_recalc_delay = ma_recalc_delay
        self.ddof = ddof
        self.anomaly_mean = None
        self.anomaly_std = None
        self.t = 0

    def time_step(self, xt):
        xt = np.reshape(xt, newshape=self.dimensions)
        ret_val = 0.
        self.buffer.append(xt)
        self.present.time_step(xt)
        if self.t >= self.buffer_len:
            pst_xt = self.buffer[0]
            self.past.time_step(pst_xt)
            if self.t >= self.present.theta + self.past.theta:
                ret_val = self.comparison_function(self.present, self.past,
                                                   self.present.alpha)
        self.ma_window.append(ret_val)
        if self.t % self.ma_recalc_delay == 0:
            self.anomaly_mean = bn.nanmean(self.ma_window)
            self.anomaly_std = bn.nanstd(self.ma_window, ddof=self.ddof)
        if self.anomaly_std is None or self.t < len(self.ma_window):
            anomaly_density = 0
        else:
            normalized_score = (ret_val - self.anomaly_mean)/self.anomaly_std
            if -4 <= normalized_score <= 4:
                anomaly_density = CDF_TABLE[round(normalized_score, 3)]
            elif normalized_score > 4:
                anomaly_density = 1.
            else:
                anomaly_density = 0.
        self.t += 1
        return ret_val, anomaly_density


@autojit(target='cpu')
def compare_models(present_model, past_model, alpha):
    tot = [0.]
    ps_w = past_model.weights
    ps_c = past_model.contexts
    for pr_x in present_model.model.node:
        pr_x_w = present_model.weights[pr_x]
        pr_x_c = present_model.contexts[pr_x]
        # dists = ne.evaluate('sum((1-alpha)*(pr_x_w - ps_w)**2 +'
        #                     '    alpha*(pr_x_c - ps_c)**2, axis=1)')
        # dists = np.add.reduce((1 - alpha)*(pr_x_w - ps_w)**2 +
        #                       alpha*(pr_x_c - ps_c)**2, axis=1)
        dists = mgng.distances(pr_x_w, ps_w, ps_c, pr_x_w, alpha)
        ps_x = np.nanargmin(dists)
        tot += dists[ps_x]
    return tot[0] / len(present_model.model.nodes())


def compare_models_w(present_model, past_model):
    tot_w = [0.]
    ps_w = past_model.weights
    for pr_x in self.present.model.nodes():
        pr_x_w = self.present.get_node(pr_x)['w']
        dists = ne.evaluate('sum((pr_x_w - ps_w)**2, axis=1)')
        ps_x = np.nanargmin(dists)
        tot_w += dists[ps_x]
    return tot_w[0] / len(self.present.model.nodes())


def compare_models_c(present_model, past_model):
    tot_c = [0.]
    ps_c = past_model.contexts
    for pr_x in self.present.model.nodes():
        pr_x_c = self.present.get_node(pr_x)['c']
        dists = ne.evaluate('sum((pr_x_c - ps_c)**2, axis=1)')
        tot_c += dists[ps_x]
    return tot_c[0] / len(self.present.model.nodes())


def main(input_file, output_file, input_frame=None,
         buffer_len=None, sampling_rate=None, index_col=None,
         skip_rows=None, ma_window=None, ma_recalc_delay=1):
    import pandas as pd
    from datetime import datetime
    if buffer_len is None:
        buffer_len = 2000
    if input_frame is None:
        signal = pd.read_csv(input_file, index_col=index_col, parse_dates=True,
                             skiprows=skip_rows)
        if sampling_rate is not None:
            signal = signal.resample(sampling_rate)
    else:
        signal = input_frame
    if ma_window is None:
        ma_window = len(signal)
    print(signal.head())
    print(signal.tail())
    print('Seting up model.')
    amgng = AMGNG(comparison_function=compare_models,
                  buffer_len=buffer_len, dimensions=signal.shape[1],
                  prest_gamma=buffer_len//2, prest_lmbda=buffer_len*6,
                  prest_theta=buffer_len, pst_gamma=buffer_len//2,
                  pst_lmbda=buffer_len*6, pst_theta=buffer_len,
                  ma_window_len=ma_window, ma_recalc_delay=ma_recalc_delay)
    scores = np.zeros(len(signal))
    pscores = np.zeros(len(signal))
    print('Processing {} rows'.format(len(signal)))
    start = datetime.now()
    for t, xt in enumerate(signal.values):
        if t % (len(signal)//100) == 0:
            print('{}% done. Sample datapoint: {}'
                  .format(t / (len(signal)//100), xt))
        scores[t], pscores[t] = amgng.time_step(xt)
    time_taken = (datetime.now() - start).total_seconds()
    print('It took {} seconds to process the signal'.format(time_taken))
    signal['anomaly_score'] = pd.Series(scores, index=signal.index)
    signal['anomaly_density'] = pd.Series(pscores, index=signal.index)
    print('Writing results to {}'.format(output_file))
    signal.to_csv(output_file)


if __name__ == '__main__':
    import sys
    args = sys.argv
    if '--input_file' in args:
        input_file = args[args.index('--input_file') + 1]
    else:
        input_file = 'samples.csv'
    if '--output_file' in args:
        output_file = args[args.index('--output_file') + 1]
    else:
        output_file = '{}_out.csv'.format(input_file)
    if '--buffer_len' in args:
        buffer_len = int(args[args.index('--buffer_len') + 1])
    else:
        buffer_len = None
    if '--sampling_rate' in args:
        sampling_rate = args[args.index('--sampling_rate') + 1]
    else:
        sampling_rate = None
    if '--index_col' in args:
        index_col = args[args.index('--index_col') + 1]
    else:
        index_col = None
    if '--skip_rows' in args:
        skip_rows = args[args.index('--skip_rows') + 1].split(',')
        skip_rows = [int(r) for r in skip_rows]
    else:
        skip_rows = None
    print(args)
    main(input_file, output_file, buffer_len, sampling_rate,
         index_col, skip_rows)
