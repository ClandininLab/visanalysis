#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:09:29 2018

@author: mhturner
"""
import pickle
def stash(obj, name, data_directory =  '/Users/mhturner/documents/stashedObjects/'):
    with open(data_directory + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def get(name ,data_directory =  '/Users/mhturner/documents/stashedObjects/'):
    with open(data_directory + name + '.pkl', 'rb') as f:
        return pickle.load(f)