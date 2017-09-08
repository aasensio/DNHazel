#!/usr/bin/env python
from __future__ import print_function
# -*- coding: iso-8859-15 -*-
# Convert string to lower case up to the first occurrence of a separator
def lower_to_sep(string, separator='='):
	line=string.partition(separator)
	string=str(line[0]).lower()+str(line[1])+str(line[2])
	return string

from configobj import ConfigObj
import sys
import os
from subprocess import call

if (len(sys.argv) < 2):
	print("You need to give the configuration file")
	exit()

print("Using configuration file = {0}".format(sys.argv[1]))

# Transform all keys to lowercase to avoid problems with
# upper/lower case
f = open(sys.argv[1],'r')
input_lines = f.readlines()
f.close()
input_lower = ['']
for l in input_lines:
	input_lower.append(lower_to_sep(l)) # Convert keys to lowercase

config = ConfigObj(input_lower)

file = open('conf.input','w')

# Write general information
file.write("'"+config['general']['type of computation']+"'\n")
file.write(config['general']['number of heliocentric angles']+'\n')

file.write('{0}\n'.format(" ".join([str(x) for x in config['general']['cosinus heliocentric angles']])))
file.write('{0}\n'.format(" ".join([str(x) for x in config['general']['weight for each heliocentric angle']])))
file.write("'"+config['general']['file with model atmosphere']+"'\n")
file.write("'"+config['general']['file with linelist']+"'\n") 
file.write("'"+config['general']['file with output results']+"'\n")

file.write(config['wavelength region']['first wavelength']+'\n')
file.write(config['wavelength region']['last wavelength']+"\n")
file.write(config['wavelength region']['wavelength step']+"\n")
file.write(config['wavelength region']['wavelength chunk size']+"\n")

	
file.close()

# Run the code
try:
	call(['mpiexec','-n',sys.argv[2],'./lte', 'conf.input'])

	#os.remove('conf.input')
except:
	#os.remove('conf.input')
	call(['pkill', '-9', 'lte'])
