########################################
#         Auxiliary functions          #
########################################

import numpy as np

######################################
#                comm                #
######################################

def comm(a,b):
    """Commutator of the matrices a,b
    
    
    Parameters
    ----------
    a : 
    b :

    Returns
    -------
    their commutator
    
    """
    return np.dot(a,b)-np.dot(b,a)

######################################
#               mykron               #
######################################

def mykron(*argv):
    """kronecker product with an arbitrary number of matrices
    
    
    Parameters
    ----------
    matrices separated by commas

    Returns
    -------
    their tensor product
    
    """

    ret=np.array([1])
    for op in argv:
        ret=np.kron(ret,op)
    return ret
