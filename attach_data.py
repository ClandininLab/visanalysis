#!/usr/bin/env python

from visanalysis import plugin
import sys
import os

experiment_file = sys.argv[1]
rigID = sys.argv[2]


if rigID == 'Bruker':
    plug = plugin.bruker.BrukerPlugin()
elif rigID == 'AODscope':
    plug = plugin.aodscope.AodScopePlugin()
else:
    plug = plugin.base.BasePlugin()

experiment_file_name = experiment_file.split('.')[0]
file_path = os.path.join(os.getcwd(), experiment_file)
data_directory = os.path.join(os.getcwd(), experiment_file_name.replace('-',''))

plug.attachData(experiment_file_name, file_path, data_directory)

print('Attached data to {}'.format(experiment_file))
