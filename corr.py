"""corr: 
compute the correlation functions in an infinite MPS

It calls routines defined in package my_package/
"""

import sys, argparse

import numpy as np

# My modules imported from the package my_package.
from my_package import functions

##########################
# Begin of the main code #
##########################

def main(cur_D):
    max_len = 60 #it is understood that the correlation length is bigger than this
    print 'Using max_len', max_len
    # Model
    model = 'Ising'
    pars = { 'J': 1.0, 'h': 1.0} #Dictionary specifying parameters of the model
#    model = 'XXZ'
#    pars = {'s':0.25}
    accuracy_conv = 1e-04

    # Initialize the Hamiltonian
    H = functions.init_H(model, pars)

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
        print 'file ', in_file_name, ' not found. Exiting.'
        sys.exit(1)
    # The wave function is
    # A1.A2.A1.A2 ... A1.A2 Lold_t B1.B2.B1.B2 ...
    # and the left transfer matrix is A1.A2, the right one B1.B2

    if model == 'Ising':
        corr_Ising(cur_D, H, model, pars, max_len, accuracy_conv, A1, A2, Lold_t, B1, B2)
    elif model == 'XXZ':
        corr_XXZ(cur_D, H, model, pars, max_len, accuracy_conv, A1, A2, Lold_t, B1, B2)
#     elif model == 'RXXZ':
#         corr_RXXZ(cur_D, H, accuracy_conv, A1, A2, Lold_t, B1, B2)
#     elif model == 'XXX_Sone':
#         corr_XXX_Sone(cur_D, H, accuracy_conv, A1, A2, Lold_t, B1, B2)
    else:
        print 'Unknown model ', model

    print ('End of the program, exiting.')

######################################
#       function: corr_Ising         #
######################################

def corr_Ising(cur_D, H, model, pars, max_len, accuracy_conv, A1, A2, Lold_t, B1, B2):
    """Compute correlators for Ising.

    Parameters
    ----------
    D :
    H :
    accuracy_conv :
    A1 :
    A2 : 
    Lold_t : 
    B1 : 
    B2:

    Returns
    -------
    nothing

    """
    
    # Preliminary: one point function is zero?
    print 'one pf sz'
    site = 1
    corrsz1 = functions.one_point_function(A1, A2, Lold_t, \
                                           H.sz, site)
    print site, corrsz1
    site = 2
    corrsz2 = functions.one_point_function(A1, A2, Lold_t, \
                                           H.sz, site)
    print site, corrsz2
    print 'one pf sx'
    site = 1
    corrsx1 = functions.one_point_function(A1, A2, Lold_t, \
                                           H.sx, site)
    print site, corrsx1
    site = 2
    corrsx2 = functions.one_point_function(A1, A2, Lold_t, \
                                           H.sx, site)
    print site, corrsx2
    print 'one pf en'
    site = 1
    corren1 = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                   H.energy_op, site)
    print site, corren1
    site = 2
    corren2 = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                   H.energy_op, site)
    print site, corren2
    print 'one pf S'
    site = 1
    corrS1 = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                H.S, site)
    print site, corrS1
    site = 2
    corrS2 = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                H.S, site)
    print site, corrS2
    print 'one pf Salt'
    site = 1
    corr = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                H.Salt, site)
    print site, corr
    site = 2
    corr = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                H.Salt, site)
    print site, corr
    print 'one pf Sav'
    site = 1
    corr = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                H.Sav, site)
    print site, corr
    site = 2
    corr = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                H.Sav, site)
    print site, corr
    print 'one pf R'
    site = 1
    corr = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                H.R, site)
    print site, corr
    site = 2
    corr = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2, \
                                                H.R, site)
    print site, corr

    # 2 Point functions
    out_file_name = 'results/idmrg_'+model+'_'
    for x in pars:
        out_file_name += x+str(pars[x])+'_'
    out_file_name += 'en_en_D'+str(cur_D)+'_eps'+str(accuracy_conv)
    f = open(out_file_name, 'w')
    f.write("#dist en-en\n")
    if model == 'Ising': 
        for i in range(5,max_len,1):
            corr = functions.two_pf_bond_op(A1, A2, Lold_t, \
                                            H.energy_op, H.energy_op, i)
            if np.mod(i, 2) == 0: # i even
                to_write = str(i)+" "+str(np.real(corr - corren1*corren2))+"\n"
            else: # i odd
                to_write = str(i)+" "+str(np.real(corr - corren1*corren1))+"\n"
            f.write(to_write)
    else:
        print 'wrong'
    f.close()
    print 'en(1)-en(i) correlator, max_len', max_len, 'in ', out_file_name

    out_file_name = 'results/idmrg_'+model+'_'
    for x in pars:
        out_file_name += x+str(pars[x])+'_'
    out_file_name += 'sz_sz_D'+str(cur_D)+'_eps'+str(accuracy_conv)
    f = open(out_file_name, 'w')
    f.write("#dist sz-sz\n")
    if model == 'Ising': 
        for i in range(5,max_len,1):
            corr = functions.two_pf_single_site_op(A1, A2, Lold_t, \
                                                       H.sz, H.sz, i)
            if np.mod(i, 2) == 0: # i even
                to_write = str(i)+" "+str(np.real(corr - corrsz1*corrsz2))+"\n"
            else: # i odd
                to_write = str(i)+" "+str(np.real(corr - corrsz1*corrsz1))+"\n"
            f.write(to_write)
    else:
        print 'wrong'
    f.close()
    print 'sz(1)-sz(i) correlator, max_len', max_len, 'in ', out_file_name

    out_file_name = 'results/idmrg_'+model+'_'
    for x in pars:
        out_file_name += x+str(pars[x])+'_'
    out_file_name += 'sx_sx_D'+str(cur_D)+'_eps'+str(accuracy_conv)
    f = open(out_file_name, 'w')
    f.write("#dist sx-sx\n")
    if model == 'Ising': 
        for i in range(5,max_len,1):
            corr = functions.two_pf_single_site_op(A1, A2, Lold_t, \
                                                       H.sx, H.sx, i)
            if np.mod(i, 2) == 0: # i even
                to_write = str(i)+" "+str(np.real(corr - corrsx1*corrsx2))+"\n"
            else: # i odd
                to_write = str(i)+" "+str(np.real(corr - corrsx1*corrsx1))+"\n"
            f.write(to_write)
    else:
        print 'wrong'
    f.close()
    print 'sx(1)-sx(i) correlator, max_len', max_len, 'in ', out_file_name

######################################
#       function: corr_XXZ           #
######################################

def corr_XXZ(D, H, model, pars, max_len, accuracy_conv, A1, A2, Lold_t, B1, B2):
    """Compute correlators for XXZ.

    Parameters
    ----------
    D :
    H :
    accuracy_conv :
    A1 :
    A2 : 
    Lold_t : 
    B1 : 
    B2:

    Returns
    -------
    nothing

    """
    # 2 Point functions
    # Op 1
    op1_name = 'D2inv'
    op1 = H.D2inv
    # vev_op1_1 = functions.six_sites_op_one_point_function(A1, A2, Lold_t, B1, B2,\
    #                                          op1, 1)
    # vev_op1_2 = functions.six_sites_op_one_point_function(A1, A2, Lold_t, B1, B2,\
    #                                          op1, 2)
    vev_op1_1 = functions.four_sites_op_one_point_function(A1, A2, Lold_t, B1, B2,\
                                             op1, 1)
    vev_op1_2 = functions.four_sites_op_one_point_function(A1, A2, Lold_t, B1, B2,\
                                             op1, 2)
    # vev_op1_1 = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2,\
    #                                          op1, 1)
    # vev_op1_2 = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2,\
    #                                          op1, 2)
    print "op1: <vac|",op1_name,"(i)|vac> = ", vev_op1_1, " i=1 ; ", vev_op1_2, " i=2 "
    # Op 2
    op2_name = 'D2inv'
    op2 = H.D2inv
    # vev_op2_1 = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2,\
    #                                          op2, 1)
    # vev_op2_2 = functions.bond_op_one_point_function(A1, A2, Lold_t, B1, B2,\
    #                                          op2, 2)
    vev_op2_1 = functions.four_sites_op_one_point_function(A1, A2, Lold_t, B1, B2,\
                                             op2, 1)
    vev_op2_2 = functions.four_sites_op_one_point_function(A1, A2, Lold_t, B1, B2,\
                                             op2, 2)
    # vev_op2_1 = functions.six_sites_op_one_point_function(A1, A2, Lold_t, B1, B2,\
    #                                          op2, 1)
    # vev_op2_2 = functions.six_sites_op_one_point_function(A1, A2, Lold_t, B1, B2,\
    #                                          op2, 2)
    print "op2: <vac|",op2_name,"(i)|vac> = ", vev_op2_1, " i=1 ; ", vev_op2_2, " i=2 "
    # Compute corr
    out_file_name = 'results/idmrg_'+model+'_'
    for x in pars:
        out_file_name += x+str(pars[x])+'_'
    out_file_name += op1_name+'_'+op2_name+'_D'+str(cur_D)+'_eps'+str(accuracy_conv)
    f = open(out_file_name, 'w')
    f.write("#dist "+op1_name+"-"+op2_name+"\n")
    in_site = 5 # Assumes only operators on 4 sites maximum
    for i in range(in_site,max_len,1):
        if op1_name == 'sz' or op1_name == 'sx':
            corr = functions.two_pf_single_site_op(A1, A2, Lold_t, op1, op2, i)
        elif op1_name == 'S1':
            corr = functions.two_pf_bond_op(A1, A2, Lold_t, op1, op2, i)
        elif op1_name == 'S2' or op1_name == 'R2' or op1_name == 'R1' or op1_name == 'D2' \
             or op1_name == 'D2inv' or op1_name == 'S2inv' :
            corr = functions.two_pf_4_sites_op(A1, A2, Lold_t, op1, op2, i)
        elif op1_name == 'S3' or op1_name == 'R3':
            corr = functions.two_pf_6_sites_op(A1, A2, Lold_t, op1, op2, i)
        else: 
            print 'Wrong operator. Exiting'
            sys.exit(1)
        # Write
        print "corr at i ", i, " = ", corr
        if np.mod(i, 2) == 0: # i even. First operator is always placed at site 1
            # subtract 1 since we print the distance 
            to_write = str(i-1)+" "+str(np.real(corr - vev_op1_1*vev_op2_2))+"\n"
        else: # i odd
            to_write = str(i-1)+" "+str(np.real(corr - vev_op1_1*vev_op2_1))+"\n"
        f.write(to_write)

    f.close()
    print op1_name+'-'+op2_name+' correlator, max_len', max_len, 'in ', out_file_name
# Return nothing


# Executed if called as a script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='corr')
    parser.add_argument('-D', '--aux', help='auxiliary dim')
    args = parser.parse_args()
    cur_D = vars(args)['aux']
    print 'Doing D = ', cur_D
    main(cur_D)

