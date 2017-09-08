#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# Convert string to lower case up to the first occurrence of a separator
def lower_to_sep(string, separator='='):
	line=string.partition(separator)
	string=str(line[0]).lower()+str(line[1])+str(line[2])
	return string

from configobj import ConfigObj
import sys
import os
import numpy as np
from subprocess import call
import glob

def run_model(conf, mus, model, output):

	ones = [1] * len(mus)

	# Transform all keys to lowercase to avoid problems with
	# upper/lower case
	f = open(conf,'r')
	input_lines = f.readlines()
	f.close()
	input_lower = ['']
	for l in input_lines:
		input_lower.append(lower_to_sep(l)) # Convert keys to lowercase

	config = ConfigObj(input_lower)

	file = open('conf.input','w')

	# Write general information
	file.write("'"+config['general']['type of computation']+"'\n")
	file.write('{0}\n'.format(len(mus)))

	file.write('{0}\n'.format(" ".join([str(x) for x in mus])))
	file.write('{0}\n'.format(" ".join([str(x) for x in ones])))
	file.write("'"+model+"'\n")
	file.write("'"+config['general']['file with linelist']+"'\n") 
	file.write("'"+output+"'\n")

	file.write(config['wavelength region']['first wavelength']+'\n')
	file.write(config['wavelength region']['last wavelength']+"\n")
	file.write(config['wavelength region']['wavelength step']+"\n")
	file.write(config['wavelength region']['wavelength chunk size']+"\n")

		
	file.close()

#	Run the code
	call(['mpiexec','-n','2','./lte'])

mus = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.02]
files = glob.glob('ATMOS/KURUCZ/T_*.model')
for f in files:
	tmp = f.split('/')[-1]
	tmp = tmp[:-5] + 'spec'
	run_model('conf.ini', mus, f, 'RESULTS/'+tmp)