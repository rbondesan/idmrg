"""#!/usr/bin/python"""

"""
idmrg: a python implementation of the idmrg code
based on the algorithm of 1008.3477v2 sec 10.

Created by Roberto Bondesan, Jan 20, 2016.  Follows 1008.3477v2,
0804.2509v1. This is the main module, other modules in package
my_package/

General remark (to be moved to the doc)
Shapes:
W : D_W,d,D_W,d
Shapes at step n:
A.shape = D',d,D
B.shape = D,d,D'
L.shape = D
psi : D,d,d,D
L : D,D_W,D
R : D_W,D,D
Step n-1
L_old =  D'
L : D',D_W,D'
R : D_W,D',D'
"""

#########################
#      Load modules     #
#########################

import sys, argparse

import numpy as np

# My modules imported from the package my_package.
from my_package import functions

def main(file):
    H, D, eps = functions.init_vars(config_file)
    #Call the main routine
    functions.minimize(H, D, eps)
    print('End of the program, exiting.')

# Executed if called as a script
if __name__ == '__main__':
    # Get from command line
    parser = argparse.ArgumentParser(description='iDMRG')
    parser.add_argument('-f', '--file', help='config file')
    args = parser.parse_args()
    config_file = vars(args)['file']
    main(config_file)
