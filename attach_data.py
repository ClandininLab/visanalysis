#!/usr/bin/env python

from visanalysis import plugin
import sys
import os

experiment_filepath = sys.argv[1]
rigID = sys.argv[2]


if rigID == 'Bruker':
    plug = plugin.bruker.BrukerPlugin()
elif rigID == 'AODscope':
    plug = plugin.aodscope.AodScopePlugin()
else:
    plug = plugin.base.BasePlugin()

experiment_file_name = os.path.split(experiment_filepath)[-1].split('.')[0]
data_directory = os.path.join(os.getcwd(), experiment_file_name.replace('-',''))

plug.attachData(experiment_file_name, experiment_filepath, data_directory)

print('Attached data to {}'.format(experiment_file))
