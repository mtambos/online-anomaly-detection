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
utils module.
@author: Mario Tambos
'''

from __future__ import division, print_function

import os

import pandas as pd
import numpy as np
import inspect
from datetime import timedelta
import shutil
import csv
import re
import seaborn as sns


def fill_annotations(df, field, anomaly_match, spans_field=None,
                     method='fill', mean=None, std=None):
    if spans_field is None:
        spans_field = field
    if method == 'fill':
        first_timestamp = None
        last_anomaly = None
        previous_timestamp = None
        for i in df.index:
            current_value = str(df.loc[i, field])
            is_anomaly_match = re.match(anomaly_match, current_value)
            if is_anomaly_match:
                if first_timestamp is not None and last_anomaly == current_value:
                    df[first_timestamp:previous_timestamp][spans_field] = last_anomaly
                first_timestamp = i
                last_anomaly = current_value
            previous_timestamp = i
    elif method == 'pad':
        for i in df.index:
            current_value = str(df.loc[i, field])
            is_anomaly_match = re.match(anomaly_match, current_value)
            if is_anomaly_match:
                pad = std*np.random.randn() + mean
                start = i - timedelta(seconds=pad/2)
                stop = i + timedelta(seconds=pad/2)
                df[start:i][spans_field] = current_value
                df[i:stop][spans_field] = current_value
    else:
        ValueError('method {} not recognized.'.format(method))



def prepare_dataset(file_path, sampling_rate_str='20L', out_file_path=None):
    file_dir = os.path.dirname(file_path)
    tmp_path = os.path.join(file_dir, 'tmp.csv')
    # Load dataset into a csv reader,
    # format column names, format timestamps
    # and save it.
    print('Formatting column names, and timestamps on file {}...'
          .format(file_path))
    with open(file_path, 'rb') as r:
        reader = csv.reader(r, quotechar="'")
        with open(tmp_path, 'wb') as w:
            writer = csv.writer(w)
            header = reader.next()
            header[0] = 'timestamp'
            header[1:] = [h.strip() for h in header[1:]]
            writer.writerow(header)
            reader.next()
            for row in reader:
                row[0] = row[0].strip('[]')
                for i, item in enumerate(row[1:], 1):
                    try:
                        row[i] = float(item)
                    except:
                        row[i] = np.nan
                writer.writerow(row) 
    shutil.move(tmp_path, file_path)

    # Load dataset into a DataFrame,
    # resample at sampling_rate_str
    # and save it.
    print('Resampling...')
    df = pd.read_csv(file_path, parse_dates=True, index_col='timestamp',
                     dayfirst=True, low_memory=False)
    df = df.resample(sampling_rate_str)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.to_csv(file_path)

    # Load dataset into a csv reader,
    # add header rows needed for HTM
    # and save it.
    print('Adding rows needed for HTM...')
    with open(file_path, 'rb') as r:
        reader = csv.reader(r)
        with open(tmp_path, 'wb') as w:
            writer = csv.writer(w)
            header = reader.next()
            writer.writerow(header)
            writer.writerow(['datetime'] + ['float' for _ in header[1:]])
            writer.writerow(['T'] + [None for _ in header[1:]])
            writer.writerows(reader) 
    shutil.move(tmp_path, file_path)
    if out_file_path is not None:
        shutil.copy(file_path, out_file_path)


def find_segment_with_most_annotations(df, field, match, segment_tdelta):
    indexer = df[field].str.match(match, na=False, as_indexer=True)
    df = df[indexer]
    d = df.index.min()
    d -= timedelta(seconds=d.second, microseconds=d.microsecond)
    anomalies = []
    while d <= df.index.max():
        init = max(d - segment_tdelta, df.index.min())
        s = slice(init, d)
        ddf = df[s]
        if len(ddf) > 0:
            anomalies.append((s, len(ddf)))
        d += timedelta(minutes=1)

    return max(anomalies, key=lambda x: x[1])


def read_annotations(file_path, columns, sampling_rate=None, quotechar='"'):    
    print('Reading annotations...')
    anndf = pd.read_csv(file_path, parse_dates=True,
                        index_col='timestamp', quotechar=quotechar)
    if sampling_rate is not None:
        tmp = []
        print('Resampling annotations...')
        for r in anndf.iterrows():
            new_index = r[0]
            new_index += timedelta(microseconds=sampling_rate -
                                                r[0].microsecond%sampling_rate)
            tmp.append([new_index] + list(r[1]))
        anndf = pd.DataFrame(tmp, columns=['timestamp', 'SampleNro', 'Type',
                                           'Sub', 'Chan', 'Num', 'Aux'])
        anndf = anndf.set_index('timestamp')
    anndf[columns] = anndf[columns].astype(np.str)
    return anndf


def _plot_annotations(annotations, df, column, ytext, ax,
                      spans_column=None, span_match='',
                      annotation_index=-2, annotate=True,
                      draw_vlines=True):
    for r in annotations.iteritems():
        x = ax.convert_xunits(r[0])
        if annotate:
            y = ax.convert_yunits(df[column][r[0]])
            ax.annotate(r[1], xy=(x, y), xytext=(x, ytext))
        if draw_vlines:
            ax.axvline(x, color='r', linewidth=0.75)
    
    if spans_column is not None:
        df['block'] = (df[spans_column].shift(1) != df[spans_column])
        df['block'] = df['block'].astype(int).cumsum()
        indexer = df[spans_column].str.match(span_match, na=False,
                                            as_indexer=True)
        g = df[indexer].reset_index().groupby([spans_column, 'block'])
        groups = g.apply(lambda x: np.array(x))
        p2 = sns.color_palette('Paired')
        for i, group in enumerate(groups):
            ax.axvspan(group[0][0], group[-1][0], color=p2[i%len(p2)],
                        alpha=0.5, label=group[0][annotation_index])


def plot_results(df, data_columns, score_column, likelihood_column,
                 match, slce=None, show_plot=True, save_plot=False,
                 cut_percentile=75, axhlines=[0.5, 0.97725, 0.999968],
                 spans_column=None, normalize_columns=False,
                 second_data_columns=None, annotate=None, draw_vlines=True):
    import matplotlib as mpl
    mpl.use('Agg')

    import pylab

    if slce is not None:
        df = df[slce]
    df = df.copy()
    m_data = np.max(df[data_columns])[0]
    m_score = np.max(df[score_column])
    indexer = df.Annotation.str.match(match, na=False, as_indexer=True)
    annotations = df.Annotation[indexer]

    if normalize_columns:
        for c in data_columns:
            min_c = df[c].min()
            max_c = df[c].max()
            df[c] = (df[c] - min_c)/(max_c - min_c)

    annotate = (spans_column is None and annotate is None) or annotate
    data_colspan = 2
    if second_data_columns is not None:
        data_colspan = 1
        if normalize_columns:
            for c in second_data_columns:
                min_c = df[c].min()
                max_c = df[c].max()
                df[c] = (df[c] - min_c)/(max_c - min_c)
    with sns.color_palette('Set2') as p:
        f = pylab.figure()
        ax1 = pylab.subplot2grid((6,1), (0, 0), rowspan=data_colspan)
        df[data_columns].plot(ax=ax1, alpha=0.7)
        ax1.set_ylabel(str(data_columns))
        _plot_annotations(annotations, df, data_columns[0], m_data, ax1,
                          annotate=annotate, draw_vlines=draw_vlines)
        pylab.legend()
        if second_data_columns is not None:
            ax11 = pylab.subplot2grid((6,1), (1, 0), sharex=ax1)
            df[second_data_columns].plot(ax=ax11, alpha=0.7)
            ax11.set_ylabel(str(second_data_columns))
            _plot_annotations(annotations, df, second_data_columns[0],
                              m_data, ax11, annotate=annotate,
                              draw_vlines=draw_vlines)
            pylab.legend()
        ax2 = pylab.subplot2grid((6,1), (2, 0), rowspan=2, sharex=ax1)
        df[likelihood_column].plot(ax=ax2, color=p[1], alpha=0.7,
                                   ylim=(0, 1.2))
        ax2.set_ylabel(likelihood_column)
        for hline in axhlines:
            ax2.axhline(hline, color='b', linewidth=0.75)
        _plot_annotations(annotations, df, data_columns[0],
                          1.1, ax2, spans_column, span_match=match,
                          annotate=annotate, draw_vlines=draw_vlines)
        pylab.legend(fancybox=True, frameon=True,
                     bbox_to_anchor=(1, 0.5), loc='center left')
        ax3 = pylab.subplot2grid((6,1), (4, 0), rowspan=2, sharex=ax1)
        scores = df[score_column]
        upper_percentile = np.percentile(scores, cut_percentile)
        scores[scores > upper_percentile] = upper_percentile
        scores.plot(ax=ax3, color=p[2], alpha=0.7)
        ax3.set_ylabel(score_column)
        _plot_annotations(annotations, df, data_columns[0], m_score, ax3,
                          annotate=annotate, draw_vlines=draw_vlines)
        pylab.legend()
    return f


def f1_score(thrs, df, col, match, annotation_column='Annotation',
             scalar=None, invert_score=False):
    detected = len(df[df[col] >thrs])
    annotation_indexer = df[annotation_column].str.match(match, na=False,
                                                 as_indexer=True)
    col_thrs = df[col] > thrs
    real = len(df[annotation_indexer])
    true_positives = len(df[col_thrs & annotation_indexer])
    false_positives = len(df[col_thrs & ~annotation_indexer])
    false_negatives = len(df[~col_thrs & annotation_indexer])
    if true_positives > 0:
        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        score = 2*(precision*recall)/(precision+recall)
    else:
        precision = 0
        recall = 0
        score = 0
    if invert_score:
        score = 1 - score
    if scalar == 'F1':
        return score
    elif scalar == 'precision':
        return precision
    elif scalar == 'recall':
        return recall
    else:
        return {'column': col, 'threshold': thrs, 'detected': detected,
                'real': real, 'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision, 'recall': recall, 'F1': score}
