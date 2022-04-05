#!/usr/bin/env python3

import sys
import os

from visanalysis.plugin import base as base_plugin

experiment_filepath = sys.argv[1]
data_directory = sys.argv[2]
rigID = sys.argv[3]


if rigID == 'Bruker':
    from visanalysis.plugin import bruker
    plug = bruker.BrukerPlugin()
    print('****Bruker plugin****')
elif rigID == 'AODscope':
    from visanalysis.plugin import aodscope
    plug = aodscope.AodScopePlugin()
    print('****AODscope plugin****')
else:
    plug = base_plugin.BasePlugin()
    print('****Unrecognized plugin name****')

experiment_file_name = os.path.split(experiment_filepath)[-1].split('.')[0]

plug.attachData(experiment_file_name, experiment_filepath, data_directory)

print('Attached data to {}'.format(experiment_filepath))
