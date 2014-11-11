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
Feeds data to a OPF model.
Based on https://github.com/numenta/nupic/tree/master/examples/opf/clients/hotgym/prediction/one_gym
'''

import os
import dateutil.parser as du_parser
import csv
import importlib

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
import cla_output as nupic_output

def create_model(params, predictedField):
    model = ModelFactory.create(params)
    model.enableInference({'predictedField': predictedField})
    return model


def process_row(row, fields, predicted_field, model, shifter,
                output_handler, counter, datetime_index=True):
    '''
    Updates the model with the row's data. row[0] should have the timestamp,
    row[1:] should correspond to fields.
    '''
    if counter % 100 == 0:
        print 'read {} lines'.format(counter)
    if datetime_index:
        index = du_parser.parse(row[0])
        input_dict = {'timestamp': index}
    else:
        index = int(row[0])
        input_dict = {'step': index}
    values = [0]*len(fields)
    for i, field in enumerate(fields, 1):
        input_dict[field] = float(row[i])
        values[i-1] = input_dict[field]
        if field == predicted_field:
            p_val = input_dict[field]
    result = model.run(input_dict)
    result = shifter.shift(result)
    prediction = result.inferences['multiStepBestPredictions'][1]
    anomalyScore = result.inferences['anomalyScore']
    output_handler.write(index, p_val, prediction, anomalyScore, values=values)


def open_input_file(input_file):
    '''
    Opens input_file and returns a file pointer to it, together with a
    csv reader and the fields inferred from the file's header
    (excepting the timestamp field).
    '''
    input_file = open(input_file, 'rb')
    csv_reader = csv.reader(input_file)
    # get column names
    fields = csv_reader.next()
    fields = [f for f in fields if f not in ('timestamp', 'step')]
    # skip header rows
    csv_reader.next()
    csv_reader.next()
    return fields, csv_reader, input_file


def prepare_run(fields, predicted_field, plot, output_name, index_name='timestamp'):
    '''
    Creates an output handler and inference shifter to use when performing
    model learning.
    '''
    if plot:
        output = nupic_output.NuPICPlotOutput(y_label=predicted_field, name=output_name)
    else:
        output = nupic_output.NuPICFileOutput(columns=fields + ['prediction'],
                                              index_name=index_name,
                                              name=output_name)
    shifter = InferenceShifter()
    return shifter, output


def run_model(model, input_file, output_name, plot, predicted_field):
    fields, csv_reader, input_file = open_input_file(input_file=input_file)
    shifter, output = prepare_run(fields, predicted_field,
                                  plot, output_name)

    counter = 0
    for row in csv_reader:
        counter += 1
        process_row(row, fields, predicted_field, model, shifter, output, counter)
    input_file.close()
    output.close()


def main(cwd, input_file, output_name, plot, predicted_field, 
         model_params=None, model_name=None):
    if model_params is None:
        sep = os.path.sep
        rel_path = os.path.relpath(cwd).replace(sep, '.')
        if model_name is None:
            package_name = '{}.model_params.model_params'.format(rel_path)
        else:
            package_name = ('{}.{}_model_params.model_params'
                            .format(rel_path, model_name))
        print 'Package name: {}'.format(package_name)
        package = importlib.import_module(package_name)
        model_params = package.MODEL_PARAMS
    model = create_model(model_params, predicted_field)
    run_model(model=model, input_file=input_file, output_name=output_name,
              plot=plot, predicted_field=predicted_field)


if __name__ == '__main__':
    import sys
    plot = False
    args = sys.argv[1:]
    if "--input_file" in args:
        input_file = index_col = args[args.index('--input_file') + 1]
    else:
        input_file = 'data.csv'
    if "--plot" in args:
        plot = True
    if "--output_name" in args:
        output_name = index_col = args[args.index('--output_name') + 1]
    else:
        output_name = 'data'
    main(input_file, output_name, plot)

