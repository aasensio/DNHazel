import numpy as np
import matplotlib.pyplot as pl
import subprocess
import pdb

files = ['hsra11.mod.atmos','emaltby11.mod.atmos','mmaltby11.mod.atmos','cool11.mod.atmos']

f = open('oldconf.ini')
lines = f.readlines()
f.close()

for f in files:
	lines[10] = "File with model atmosphere = 'ATMOS/semi-empirical/{0}'\n".format(f)
	lines[11] = "File with linelist = 'DATA/kurucz_expanded.list'\n"
	lines[12] = "File with output results = 'RESULTS/{0}.spec'\n".format(f)
	
	ff = open('conf.ini','w')
	ff.writelines(lines)
	ff.close()
	
	subprocess.call(["./run.py","conf.ini","3"])
	
	lines[10] = "File with model atmosphere = 'ATMOS/semi-empirical/{0}'\n".format(f)
	lines[11] = "File with linelist = 'DATA/kurucz_plus_tio.list'\n"
	lines[12] = "File with output results = 'RESULTS/{0}.tio.spec'\n".format(f)
	
	ff = open('conf.ini','w')
	ff.writelines(lines)
	ff.close()
	
	subprocess.call(["./run.py","conf.ini","3"])