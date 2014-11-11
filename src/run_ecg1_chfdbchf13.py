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
Experiment on multidimensional ECG using
the chfdb/chf13 dataset from PhysioNet.
@author: Mario Tambos
'''
from __future__ import division, print_function

import os

import pandas as pd
import numpy as np
import inspect
from datetime import timedelta, datetime

import utils

def main(cwd, do_amgng, amgng_file, ma_window, ma_recalc_delay,
         do_cla, cla_file, buffer_len, plot):
    values = inspect.getargvalues(inspect.currentframe())[3]
    print('using parameters: {}'.format(values))
    annotations_path = os.path.join(cwd, 'annotations.csv')
    anndf = utils.read_annotations(annotations_path, ['Type'], 20000)
    amgng_df = None
    if do_amgng:
        from mgng.amgng import main as amgng_main
        print('Training AMGNG model...')
        out_file = os.path.join(cwd, 'out_amgng_{}'.format(amgng_file))
        full_path = os.path.join(cwd, amgng_file)
        start = datetime.now()
        amgng_main(input_file=full_path, output_file=out_file,
                   buffer_len=buffer_len, index_col='timestamp',
                   skip_rows=[1,2], ma_window=ma_window,
                   ma_recalc_delay=ma_recalc_delay)
        amgng_time = datetime.now() - start

        print('Reading results...')
        amgng_df = pd.read_csv(out_file, parse_dates=True,
                               index_col='timestamp')
        amgng_df['Annotation'] = anndf.Type
        print('Writing annotated results...')
        amgng_df.to_csv(out_file)
        if plot:
            utils.plot_results(amgng_df, ['ECG1'], 'anomaly_score',
                               'anomaly_density', '[rs]')
        print('Time taken: amgng={}'.format(amgng_time))

    cla_df = None
    if do_cla:
        from cla.swarm import swarm
        from cla.cla import main as cla_main
        out_file = os.path.join(cwd, 'out_cla_{}'.format(cla_file))
        print('Training CLA model...')
        full_path = os.path.join(cwd, cla_file)
        SWARM_DESCRIPTION = {
            'includedFields': [
                {
                    'fieldName': 'timestamp',
                    'fieldType': 'datetime',
                },
                {
                    'fieldName': 'ECG1',
                    'fieldType': 'float',
                },
            ],
            'streamDef': {
                'info': 'chfdbchf13 ECG1',
                'version': 1,
                'streams': [
                    {
                        'info': 'chfdbchf13',
                        'source': full_path,
                        'columns': ['*']
                    }
                ]
            },
            'inferenceType': 'TemporalAnomaly',
            'inferenceArgs': {
                'predictionSteps': [1],
                'predictedField': 'ECG1'
            },
            'iterationCount': buffer_len,
            'swarmSize': 'small'
        }
        start = datetime.now()
        swarm(cwd=cwd, input_file=cla_file,
              swarm_description=SWARM_DESCRIPTION)
        swarm_time = datetime.now() - start
        start = datetime.now()
        cla_main(cwd=cwd, input_file=full_path, output_name=out_file, plot=False,
                 predicted_field='ECG1')
        cla_time = datetime.now() - start

        print('Reading results...')
        cla_df = pd.read_csv(out_file, parse_dates=True, index_col='timestamp')
        cla_df['Annotation'] = anndf.Type
        print('Writing annotated results...')
        cla_df.to_csv(out_file)
        if plot:
            utils.plot_results(cla_df, ['ECG1'], 'anomaly_score',
                               'anomaly_likelihood', '[rs]')
        print('Time taken: swarm={}, cla={}'.format(swarm_time, cla_time))
    return amgng_df, cla_df

if __name__ == '__main__':
    import sys
    args = sys.argv
    if '--do_amgng' in args:
        do_amgng = True
    else:
        do_amgng = False
    if '--amgng_file' in args:
        amgng_file = args[args.index('--amgng_file') + 1]
    else:
        amgng_file = 'experiments/ecg1_chfdbchf13/chfdbchf13_final.csv'
    if '--do_cla' in args:
        do_cla = True
    else:
        do_cla = False
    if '--cla_file' in args:
        cla_file = args[args.index('--cla_file') + 1]
    else:
        cla_file = 'experiments/ecg1_chfdbchf13/chfdbchf13_final.csv'
    if '--cwd' in args:
        cwd = args[args.index('--cwd') + 1]
    else:
        if do_amgng:
            cwd = os.path.dirname(amgng_file)
            amgng_file = os.path.basename(amgng_file)
        elif do_cla:
            cwd = os.path.dirname(cla_file)
            cla_file = os.path.basename(cla_file)
        else:
            cwd = os.getcwd()
    if '--buffer_len' in args:
        buffer_len = int(args[args.index('--buffer_len') + 1])
    else:
        buffer_len = 300
    if '--ma_window' in args:
        ma_window = int(args[args.index('--ma_window') + 1])
    else:
        ma_window = 300
    if '--ma_recalc_delay' in args:
        ma_recalc_delay = int(args[args.index('--ma_recalc_delay') + 1])
    else:
        ma_recalc_delay = 1
    if '--plot' in args:
        plot = True
    else:
        plot = False
    main(cwd=cwd, do_amgng=do_amgng, amgng_file=amgng_file,
         ma_window=ma_window, do_cla=do_cla, cla_file=cla_file,
         buffer_len=buffer_len, plot=plot, ma_recalc_delay=ma_recalc_delay)
