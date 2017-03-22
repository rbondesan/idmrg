"""EE_vs_D: 
compute the dependence of the entanglement entropy
vs log(xi), where xi is the correlation length,
related to the second largest eigenvalue of the MPS
transfer matrix, at a given D. Then fit to extract
the central charge.

It calls routines defined in package my_package/
"""

import sys

import numpy as np

# My modules imported from the package my_package.
from my_package import functions

def main():
    # Lists
    D_list = range(25,26,5) 
    logxi_list = []
    EE_list = []
    # Model: comment one or the other to choose
    model = 'Ising'; d=2
    pars = { 'J': 1.0, 'h': 1.0} #Dictionary specifying parameters of the model
#     model = 'XXZ'; d = 2
#     pars={'s':0.25}
    mytype = float
    accuracy_conv = 1e-04

    for cur_D in D_list:
        print "cur_D ", cur_D
        # read data
        in_file_name = 'init/idmrg_'+model+'_D'+str(cur_D)+'_'
        for x in pars:
            in_file_name += x+str(pars[x])+'_'
        in_file_name += 'eps'+str(accuracy_conv)
        try:
            f = open(in_file_name, 'r')
            # Use numpy routines to read the MPS data from file
            buff = np.load(f)
            buff = np.load(f)
            buff = np.load(f)
            buff = np.load(f)
            buff = np.load(f)
            buff = np.load(f)
            A1 = np.load(f)
            A2 = np.load(f)
            Lold_t = np.load(f)
            B1 = np.load(f)
            B2 = np.load(f)
            f.close()
            print 'read from file', in_file_name
        except IOError:
            print 'IOError: file ', in_file_name, ' not found. Exiting.'
            sys.exit(1)
        # The wave function is
        # A1.A2.A1.A2 ... A1.A2 Lold_t B1.B2.B1.B2 ...
        # and the left transfer matrix is A1.A2, the right one B1.B2
        # Compute first 16 corr lengths, EE, bond energy 
        E = functions.transfer_operator(A1, A2, np.eye(d), np.eye(d))
        cur_corr_lens = functions.compute_corr_lengths(E,16)
        print "cur_corr_lens ", cur_corr_lens
        cur_xi = max(np.real(cur_corr_lens[0:-1])) #up to the last which is infty.
        print "cur_xi ", cur_xi
        # Compute EE
        cur_EE = functions.compute_ent_entropy_mixed_repr(Lold_t)
        print "log(xi) vs EE ", np.log(cur_xi), cur_EE
        # And append to lists
        logxi_list.append(np.log(cur_xi))
        EE_list.append(cur_EE)
        # Print the bond energy as a check
        H = functions.init_H(model, pars)
        En_AB = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, \
                                                     B2, H.h_bond, 1)
        En_BA = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, \
                                                     B2, H.h_bond, 2)
        print "Bond energy: ", np.mean([En_AB,En_BA]), ". Exact: ", H.hinfty
        print "Delta bond energy: ", np.mean([En_AB,En_BA]) - H.hinfty
        # Go to next D

    # Print to file to analyze
    if len(D_list) == 1:
        out_file_name = 'results/idmrg_logxi_vs_EE_'+model+'_D'+str(D_list[0])+'_'
    else:
        out_file_name = 'results/idmrg_logxi_vs_EE_'+model+'_D'+str(D_list[0])+\
            '_'+str(D_list[-1])+'_'
    for x in pars:
        out_file_name += x+str(pars[x])+'_'
    out_file_name += 'eps'+str(accuracy_conv)
    f = open(out_file_name, 'w')
    n = 0
    for cur_D in D_list:
        f.write(str(logxi_list[n])+" "+str(EE_list[n])+"\n")
        n = n + 1

    f.close()
    print 'saved to file ', out_file_name

    print ('End of the program, exiting.')

# Executed if called as a script
if __name__ == '__main__':
    main()
