import sys
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh,LinearOperator,eigs

from my_package import Ising
from my_package import RXXZ
from my_package import XXZ
from my_package import XXX_Sone
from my_package import IK

######################################
#       function: init_vars          #
######################################

def init_vars(config_file):
    """Initialize the variables. Returns the argument to be fed into
    minimize. The structure of the config file is
    D=value
    eps=value
    model=value
    coupling_name=value
    coupling_name=value
    etc

    Parameters
    ----------
    config_file

    Returns
    -------
    H :
    D :
    eps :

    """
    
    #open and read file
    with open(config_file, 'r') as f:
        [name,value] = f.readline().split("=")
        D = int(value)
        [name,value] = f.readline().split("=")
        eps = float(value)
        [name,value] = f.readline().split("=")
        if value[-1] == '\n':
            model = value[:-1] #to get rid of \n
        else:
            model = value
        #The rest is the couplings. Finish till the end of the 
        pars = {}
        for line in f:
            [name,value] = line.split("=")
            pars[name] = float(value)

    #Print data read
    print ('In init_vars: model', model, 'Parameters: ')
    for x in pars:
        print (x),'=',pars[x]
    print ("trunc_D", D, "accuracy conv", eps)

    #Here file ended and closed. Now init the Hamiltonian class.
    H = init_H(model, pars)

    return H, D, eps

######################################
#       function: init_H             #
######################################

def init_H(model, pars):
    """Initialize the Hamiltonian. Check if the model exists and returns
    an object of the respective class.

    Parameters
    ----------
    model : string specifying the name of the model
    pars :

    Returns
    -------
    H :

    """
    
    if model == 'Ising':
        H = Ising.Ising_H(pars)
        print "Initialized Ising H"
    elif model == 'Potts':
        print "Not implemented yet"
    elif model == 'RXXZ':
        H = RXXZ.RXXZ_H(pars)
        print "Initialized RXXZ H"
    elif model == 'XXZ':
        H = XXZ.XXZ_H(pars)
        print "Initialized XXZ H"
    elif model == 'XXX_Sone':
        H = XXX_Sone.XXX_Sone_H(pars)
        print "Initialized XXX_Sone"
    elif model == 'IK':
        H = IK.IK_H(pars)
        print "Initialized IK"
    else:
        print >> sys.stderr, 'In init_H: Wrong model in config file', \
            config_file, ". Exiting"
        sys.exit(1)

    return H

######################################
#       function: minimize           #
######################################

def minimize(H, trunc_D, accuracy_conv):
    """Minimize H to find the gs energy with prescribed bond dimension
    trunc_D and with accuracy accuracy_conv. Returns A1, A2,
    Lambda_old of the orthogonalized (left) unit cell in the
    thermodynamic limit and the central matrix Lambda_old.

    Parameters
    ----------
    H :
    trunc_D :
    accuracy_conv :

    Returns 
    -------
    Nothing. Saves data on file for future usage 
    (init next size and compute EE and corr functions)

    """
    max_it = 1000000 # Maximum number of iterations of the idmrg procedure

    print('In functions.minimize:')
    # Call the function initialize.
    A, Lambda, B, Lambda_old, Lenv, Renv \
        = init_idmrg(H, trunc_D, accuracy_conv)

    # Initially, try to converge fidelity
    conv_mode = 'fidelity'
    if conv_mode == 'fidelity':
        small_num = 1e-18
        if accuracy_conv < small_num:
            small_num = accuracy_conv
        Fprev = 1
    # Used in check convergence:
    if H.model == 'RXXZ' or H.model == 'XXZ':
        dn_check_conv = 2
        #check every 2 steps due to even/odd effects in XXZ chains
        #clear in XXX
    else:
        dn_check_conv = 1
                
    # Compute initial En, S
    eta, X, Y = compute_eta_X_Y(A, Lambda, B, Lambda_old, H.dtype)
    En_prev, S_prev = bond_en_and_S(H.h_bond, A, Lambda, B, \
                                            Lambda_old, eta, X, Y)
    print 'Initial En, S', En_prev, S_prev

    print ('Initialization done, recursion starts')

    # Start recursion
    is_conv = False
    for n in range(max_it):
        # check_norm(A, B, Lambda)
        
        # Compute ansatz given to eigensolver as initial state:
        if conv_mode == 'fidelity': 
            A_new, Lambda_R, Lambda_L, B_new = rotate_center(A,Lambda,B)
            psi_guess = compute_psi_guess_alt(A_new,Lambda_R,\
                                                  Lambda_old,Lambda_L,B_new)
        else:
            psi_guess = compute_psi_guess(A, B, Lambda, Lambda_old) 

        # Create proj hamiltonian as linear operator (It depends on L,R,W.)
        # Update L,R first
        Lenv, Renv = update_environments(Lenv, Renv, H.W, A, B) 

        # Create linear operator with its methods matvec.
        # Compute LWWR once and do not use the matvec routine, faster
        # but more memory usage.
        # proj_h = proj_h_mat(Lenv, Renv, H.W)
        # D = Lenv.shape[0]; dim_proj_h = (D**2)*(H.phys_dim**2)
        # proj_h = proj_h.reshape(dim_proj_h,dim_proj_h)
        # Alternative: use matvec routine, better for huge matrices
        # Create linear operator with its methods matvec        
        mv = lambda x: proj_h_vec_mul(x, Lenv, Renv, H.W)
        D = Lenv.shape[0]; dim_proj_h = (D**2)*(H.phys_dim**2)
        proj_h = LinearOperator((dim_proj_h,dim_proj_h), matvec=mv, dtype=H.dtype)

        # Compute the minimum of the current projected Hamiltonian
        # E0 is the ground state energy at size 4 + 2*n
        if H.is_hermitian == False:
            E0,psi = eigs(proj_h, k=1, which='SR', return_eigenvectors=True, \
                           v0 = psi_guess.reshape(dim_proj_h))
        else:
            E0,psi = eigsh(proj_h, k=1, which='SA', return_eigenvectors=True, \
                           v0 = psi_guess.reshape(dim_proj_h))

        # Truncation and update A,B,Lambda,Lamda_old
        Lambda_old = Lambda
        A, Lambda, B, err = truncate_and_update(psi, D, H.phys_dim, trunc_D)

        # Check convergence. TODO: write a separate function
        if conv_mode == 'fidelity':
            if np.mod(n, dn_check_conv) == 0 and n>100: #100 iterations, mininum for each D
                F = f_ortho(Lambda_R, np.diag(Lambda))
                # Check convergence: if D very small it will never converge
                # with a very small epsilon
                if 1-F < accuracy_conv:
                    # F converged
                    print ("fidelity converged at step", n, "with accuracy of", \
                               accuracy_conv)
                    print ("1-F = ", 1-F)
                    # Save data to file
                    save_to_file(H, trunc_D, accuracy_conv, A, Lambda, B, \
                                 Lambda_old, Lenv, Renv, \
                                 0, np.array([0]), np.array([0]))
                    # Now try to converge En, S. If it takes too much time,
                    # it will be stopped by hand. However, matrices saved.
                    print "Now try to converge En, S"
                    n_conv_F = n
                    conv_mode = 'En,S'
                elif np.absolute(F-Fprev) < small_num and F > small_num: 
                    # beginning F = 0 for several iterations. The or takes care
                    # of an even/odd effect observed in some cases
                    print 'Stuck, step', n, \
                        'no convergence to exact value of F: 1.','F', F,\
                        'but Delta F <',small_num
                    # Save data to file
                    save_to_file(H, trunc_D, accuracy_conv, A, Lambda, B, \
                                 Lambda_old, Lenv, Renv, \
                                 0, np.array([0]), np.array([0]))
                    # Now try to converge En, S. If it takes too much time,
                    # it will be stopped by hand. However, matrices saved.
                    print "Now try to converge En, S"
                    n_conv_F = n
                    conv_mode = 'En,S'
                else:
                    Fprev = F
        else: #conv_mode != fidelity 
            if np.mod(n, dn_check_conv) == 0 and n>100:
                eta, X, Y = compute_eta_X_Y(A, Lambda, B, \
                                                Lambda_old, H.dtype)
                En, S = bond_en_and_S(H.h_bond, A, Lambda, B, \
                                          Lambda_old, eta, X, Y)

                print n," ",np.real(En)," ",S
                # Check convergence
                if np.absolute(En-En_prev) < accuracy_conv and \
                   np.absolute(S-S_prev) < accuracy_conv:
                    print 'En,S converged after', n, 'iterations.'
                    is_conv = True
                    break
                else:
                    En_prev = En
                    S_prev = S
                    if np.mod(n-n_conv_F, 100) == 0:
                        print 'dump current matrices'
                        save_to_file(H, trunc_D, accuracy_conv, A, Lambda, B, \
                                     Lambda_old, Lenv, Renv, eta, X, Y)
        
        # Else, continue, here counter n+1->n, until n reaches max_it
        # End of the loop

    if is_conv == False:
        print ("Reached max_it value ", max_it, " and failed to converge "\
               "with accuracy ", accuracy_conv)
        if conv_mode == 'fidelity':
            print ("1-fidelity", 1-f_ortho(Lambda_R, np.diag(Lambda)))
               
    ######################################################
    # At this point the wave function is
    # ...A^{si}LB^{si+1}Lold^{-1}A^{si+2}LB^{si+3}Lold^{-1}... 
    # Orthonormalize:
    # A1.A2.A1.A2 ... A1.A2 Lold_t B1.B2.B1.B2 ...
    # and the left transfer matrix is A1.A2, the right one B1.B2
    # If arrives here, it means that the convergence of En, S has
    # succeded. Replace the file previously saved
    save_to_file(H, trunc_D, accuracy_conv, A, Lambda, B, \
                 Lambda_old, Lenv, Renv, 0, np.array([0]), np.array([0]))

#   Do not return anything

######################################
#      function: save_to_file        #
######################################

def save_to_file(H, trunc_D, accuracy_conv, A, Lambda, B, Lambda_old, \
                 L, R, eta, X, Y):
    """Dump to file the matrices computed in minimize

    Parameters
    ----------
    H :
    trunc_D : 
    accuracy_conv :
    A : 
    Lambda : 
    B : 
    Lambda_old : 
    L :
    R :
    eta :
    X :
    Y :

    Returns
    -------
    None

    """

    # First compute A1,A2,Lold_t,B1,B2
    if X.size == 1 or Y.size == 1:
        # means one has to compute again X,Y,eta   
        eta, X, Y = compute_eta_X_Y(A, Lambda, B, Lambda_old, H.dtype)
    else:
        # do nothing, use matrices give.
        pass
    A1, A2, Lambda_old_t, B1, B2 = mixed_repr_unit_cell(A, \
                                   Lambda/np.sqrt(eta), B, Lambda_old, X, Y)
    # Save data A, Lambda, B, Lambda_old, Lenv, Renv, to start next D
    # recurrence from it, and A1,A2,Lambda_old_t,B1,B2 to use for
    # entropy, correlators etc
    file_name = 'init/idmrg_'+H.model+'_D'+str(trunc_D)+'_'
    for x in H.pars:
        file_name += x+str(H.pars[x])+'_'
    file_name += 'eps'+str(accuracy_conv)
    f = open(file_name, 'w')
    # Use numpy routines
    np.save(f, A)
    np.save(f, Lambda)
    np.save(f, B)
    np.save(f, Lambda_old)
    np.save(f, L)
    np.save(f, R)
    np.save(f, A1)
    np.save(f, A2)
    np.save(f, Lambda_old_t)
    np.save(f, B1)
    np.save(f, B2)
    print 'save_to_file: saved to ', file_name
    f.close()
    # end of the routine

######################################
#      function: init_idmrg          #
######################################

def init_idmrg(H, trunc_D, accuracy_conv):
    """init idmrg either from file or by computing
    2 and 4 spin problem by diagonalizing exactly h, calling init_idmrg_4

    Parameters
    ----------
    H : 
    trunc_D :
    accuracy_conv :

    Returns
    -------
    In this order:
    curA , curB , curLambda , oldLambda , curL , curR 

    """
    # Open file previous trunc_D and previous eps
    min_D = 4
    for cur_D in range(trunc_D, min_D - 1, -1):
        cureps = accuracy_conv
        while cureps < 1e-05:
            in_file_name = 'init/idmrg_'+H.model+'_D'+str(cur_D)+'_'
            for x in H.pars:
                in_file_name += x+str(H.pars[x])+'_'
            in_file_name += 'eps'+str(cureps)
            try:
                f = open(in_file_name, 'r')
                # Use numpy routines
                curA = np.load(f)
                curLambda = np.load(f)
                curB = np.load(f)
                oldLambda = np.load(f)
                curL = np.load(f)
                curR = np.load(f)
                f.close()
                print 'init_idmrg: from file', in_file_name
                return curA, curLambda, curB, oldLambda, curL, curR
            except IOError:
                #print 'init_idmrg: file ', in_file_name, ' not found.'
                pass

            cureps *= 10

    # If here it means that no file has been found, and curD = min_D
    # so init_idmrg_4 is called
    print 'init from 4 sites'
    curA, curLambda, curB, oldLambda, curL, curR \
        = init_idmrg_4(H.h_two_sites, H.phys_dim, H.W, \
                       H.vl, H.vr, trunc_D, H.is_hermitian)

    return curA, curLambda, curB, oldLambda, curL, curR

######################################
#      function: init_idmrg_4        #
######################################

def init_idmrg_4(h, d, W, v_left, v_right, trunc_D, is_hermitian):
    """Computes 2 and 4 spin problem by diagonalizing exactly h

    Parameters
    ----------
    h : 2x2 matrix to be diagonalized 
        (h is expressed in the basis |i1>\otimes |i2> => |d*i1 + i2>, i=0,...,d-1)
    d : phys dim
    W : MPO
    v_left : left initial vector: H is v_left W ...
    v_right : right initial vector: H is  ... W v_right

    Returns
    -------
    In this order:
    curA , curB , curLambda , oldLambda , curL , curR 
    After 1-th step, D=d^2, D'=d.

    """

    # Checks of the dimension of h: TODO
    D = 1
    # Calls the eigensolver eig for hermitian complex matrices or symm
    # real matrix
    if is_hermitian == False:
        la,v = linalg.eig(h)
    else:
        la,v = linalg.eigh(h)
    index_min = np.where(la==min(la))[0][0]
    psi = v[:,index_min]
    print "2 sites: gs en =", la[index_min]

    # Perform a SVD after reshaping. if psi.shape == (d D_L, d
    # D_R), then when full_matrices=False it returns U.shape = (d D_L,
    # D'), Vh.shape = (D', d D_R), with D' = min(d D_L, d D_R). Here
    # D_L = D_R = 1.  Better to use our routine than plain svd since
    # there might be zero or very small singular values and we want to
    # get rid of them (later we need to invert lambda_old).
    curA, curLambda, curB, err = truncate_and_update(psi, D, d, trunc_D)
    # and initialize A,B,Lambda,Lambda_old. Here not need to normalize, since 
    # s already normalized by definition, since psi is a normalized wave function
    # Reshape of curA and curB as matrices
    curA = curA.reshape(curA.shape[1],curA.shape[2])
    curB = curB.reshape(curB.shape[0],curB.shape[1])
    oldLambda = np.array([1])

    # Now 4 sites: compute the projected Hamiltonian.
    # Update environments with A, B (not usual routine)
    curL = np.tensordot(v_left, np.conj(curA), axes=(2,0))
    curL = np.tensordot(curA, curL, axes=(0,0))
    curR = np.tensordot(v_right, np.conj(curB), axes=(2,1))
    curR = np.tensordot(curB, curR, axes=(1,1))
    curR = curR.transpose(1,0,2)
    h_proj = proj_h_mat(curL, curR, W)
    D = curL.shape[0]
    dim = D*d
    if is_hermitian == False:
        la,v = linalg.eig(h_proj.reshape(dim*dim,dim*dim)) 
    else:
        la,v = linalg.eigh(h_proj.reshape(dim*dim,dim*dim))
    index_min = np.where(la==min(la))[0][0]
    psi = v[:,index_min]  # Ground state wf for 4 sites, it is a vector of dim d^2
    # Assign the new matrices
    oldLambda = curLambda
    curA, curLambda, curB, err = truncate_and_update(psi, D, d, trunc_D)
    print "4 sites: gs en =", la[index_min]

    return curA, curLambda, curB, oldLambda, curL, curR

######################################
#      function: init_itebd          #
######################################

def init_itebd(h, d, W, v_left, v_right, trunc_D):
    """Computes 2 and 4 spin problem by diagonalizing exactly h
    and returns GA,LA,GB,LB. 

    Parameters
    ----------
    h : 2x2 matrix to be diagonalized 
        (h is expressed in the basis |i1>\otimes |i2> => |d*i1 + i2>, i=0,...,d-1)
    d : phys dim
    W : MPO
    v_left : left initial vector: H is v_left W ...
    v_right : right initial vector: H is  ... W v_right

    Returns
    -------
    In this order:
    curA , curB , curLambda , oldLambda , curL , curR 
    After 1-th step, D=d^2, D'=d.

    Later to be done without MPO and vl,vr...

    """
    # Checks of the dimension of h: TODO
    Dp = 1
    D = d

    # L,R: first initialize them at the 0-th step even though not used
    # below, but do it for consitency. As customary (see eg (16) of 0804.2509):
    D_W = v_left.shape[0]
    curL = np.zeros([Dp, D_W, Dp]); curL[0,:,0] = v_left
    curR = np.zeros([D_W, Dp, Dp]); curR[:,0,0] = v_right

    # Calls the eigensolver eig for hermitian complex matrices or symm
    # real matrix
    print h
    la,v = linalg.eigh(h)
    psi = v[:,0]  # Ground state wf for 2 sites, it is a vector of dim d^2
    print "2 sites: E =", la[0], " , gs: ", psi

    # Perform a SVD after reshaping
    psi = psi.reshape((d, d))
    # SVD: if psi.shape == (d D_L, d D_R), then when full_matrices=False 
    # it returns U.shape = (d D_L, D'), Vh.shape = (D', d D_R), with 
    # D' = min(d D_L, d D_R). Here D_L = D_R = 1.
    U, s, Vh = linalg.svd(psi, full_matrices=False)
    # and initialize A,B,Lambda,Lambda_old. Here not need to normalize, since 
    # s already normalized by definition, since psi is a normalized wave function
    curA = U.reshape((1,d,D))
    curLambda = s
    curB = Vh.reshape((D,d,1))
    oldLambda = np.array([1])

    # Now 4 sites: compute the projected Hamiltonian
    curL, curR = update_environments(curL, curR, W, curA, curB) 
    dim = D*d
    h_proj = proj_h_mat(curL, curR, W)

    la,v = linalg.eigh(h_proj.reshape(dim*dim,dim*dim))
    psi = v[:,0]  # Ground state wf for 4 sites, it is a vector of dim d^2
    print "4 sites: E =", la[0], " , gs: "
    print psi.reshape(1,dim*dim)
    # Assign the new matrices
    oldLambda = curLambda
    curA, curLambda, curB, err = truncate_and_update(psi, D, d, trunc_D)

    # Define the matrices to be returned:
    curGA = np.tensordot(np.diag(1/oldLambda),curA,axes=(1,0))
    curLA = curLambda
    curGB = np.tensordot(curB,np.diag(1/oldLambda),axes=(2,0))
    curLB = Lambdaold

    return curGA, curLA, curGB, curLB

######################################
#   function: update environments    #
######################################

def update_environments(my_L, my_R, my_W, my_A, my_B):
    """Update environments:

    Parameters
    ----------
    my_L : current left environment L: 
    my_R : current right environment R: 
    my_W : MPO 
    my_A : current A: 
    my_B : current A: 

    Returns
    -------
    In this order:
    curL , curR 

    """

    new_L = update_L_environment(my_L, my_W, my_A)
    new_R = update_R_environment(my_R, my_W, my_B)

    return new_L, new_R

######################################
#   function: update_L_environment   #
######################################

def update_L_environment(my_L, my_W, my_A):
    """Update left environment: splits computation
    in successive contractions (matrix multiplications)
    to reduce the complexity. 

    Parameters
    ----------
    my_L : current left environment, shape = D',D_W,D'
    my_W : MPO
    my_A : A

    Returns
    -------
    In this order:
    res : updated L, shape = D,D_W,D

    """

    # Perform the first contraction: my_L with my_W
    res = np.tensordot(my_L, my_W, axes=(1,0))
    # Second contraction: Astar with res
    Astar = np.conjugate(my_A)
    res = np.tensordot(Astar, res, axes=([0,1], [0,2]))
    # Last contraction: res2 with curA
    res = np.tensordot(res, my_A, axes=([1,3],[0,1]))
    # Already canonical form

    return res

######################################
#   function: update_R_environment   #
######################################

def update_R_environment(my_R, my_W, my_B):
    """Update right environment: splits computation in successive
    contractions (matrix multiplications) to reduce the
    complexity. See description of update_L_environment 

    Parameters
    ----------
    my_R : current right environment R
    my_W : MPO
    my_B : B

    Returns
    -------
    In this order:
    res : updated R

    """

    # Contraction: Bstar with my_R
    Bstar = np.conjugate(my_B)
    res = np.tensordot(Bstar, my_R, axes=(2,1))
    # Contraction: W, res
    res = np.tensordot(my_W, res, axes=([1,2],[1,2]))
    # Contraction: res, B
    # Last contraction: res with curB
    res = np.tensordot(res, my_B, axes=([1,3],[1,2]))
    # Already canonical form

    return res

######################################
#    function: compute_psi_guess     #
######################################
def compute_psi_guess(my_A, my_B, my_Lambda, my_Lambda_old):
    """Computes psi guess, use the guess Eq. 338 of Schwollock review
    ('10). Do a series of multiplications splitting contractions.

    Parameters
    ----------
    my_A : 
    my_B :
    my_Lambda :
    my_Lambda_old :

    Returns
    -------
    psi : psi_guess

    """
    res = np.tensordot(np.diag(my_Lambda), my_B, axes=(1,0))
    if my_Lambda_old.ndim == 1:
        res = np.tensordot(res, np.diag(1/my_Lambda_old), axes=(2,0))
    elif my_Lambda_old.ndim == 2:
        res = np.tensordot(res, linalg.inv(my_Lambda_old), axes=(2,0))
    else:
        print ("ERROR: in left_normalize_unit_cell, my_Lambda_old has wrong ndim")
    res = np.tensordot(res, my_A, axes=(2,0))
    psi = np.tensordot(res, np.diag(my_Lambda), axes=(3,0))
    # psi.shape = (D,d,d,D), ok.

    return psi

######################################
#  function: compute_psi_guess_alt   #
######################################
def compute_psi_guess_alt(my_A_new, my_Lambda_R, my_Lambda_old, my_Lambda_L,\
                          my_B_new):
    """Computes psi guess, starting vector for the minimization
    (diagonalization) procedure. Use the guess Eq. 339-340 of Schwollock
    review ('10). Do a series of multiplications splitting contractions.

    Parameters
    ----------
    my_A_new : 
    my_Lambda_R :
    my_Lambda_old :
    my_Lambda_L :
    my_B_new :

    Returns
    -------
    psi : psi_guess

    """
    D_new, d, D = my_B_new.shape 
    Dp = my_Lambda_old.size

    # Note: by definition my_Lambda_old is invertible since small
    # singular values have been truncated.
    res = np.tensordot(my_A_new,my_Lambda_R, axes=(2,0))
    res = np.tensordot(res, np.diag(1/my_Lambda_old), axes=(2,0))
    res = np.tensordot(res, my_Lambda_L, axes=(2,0))
    psi = np.tensordot(res, my_B_new, axes=(2,0))
    # psi.shape = (D,d,d,D), ok.

    return psi

######################################
#   function: compute_psi_guess_AB   #
######################################
def compute_psi_guess_AB(my_GA, my_LA, my_GB, my_LB):
    """Computes psi guess of cell AB

    Parameters
    ----------
    my_GA : 
    my_LA : 
    my_GB : 
    my_LB : 

    Returns
    -------
    psi : psi_guess

    """

    res = np.tensordot(np.diag(my_LB), my_GA, axes=(1,0))
    res = np.tensordot(res, np.diag(my_LA), axes=(2,0))
    res = np.tensordot(res, my_GB, axes=(2,0))
    psi = np.tensordot(res, np.diag(my_LB), axes=(3,0))
    # psi2.shape = (Dp,d,d,Dp), ok.

    return psi

######################################
#   function: compute_psi_guess_BA   #
######################################
def compute_psi_guess_BA(my_GA, my_LA, my_GB, my_LB):
    """Computes psi guess of cell BA

    Parameters
    ----------
    my_GA : 
    my_LA : 
    my_GB : 
    my_LB : 

    Returns
    -------
    psi : psi_guess

    """

    res = np.tensordot(np.diag(my_LA), my_GB, axes=(1,0))
    res = np.tensordot(res, np.diag(my_LB), axes=(2,0))
    res = np.tensordot(res, my_GA, axes=(2,0))
    psi = np.tensordot(res, np.diag(my_LA), axes=(3,0))
    # psi2.shape = (Dp,d,d,Dp), ok.

    return psi


######################################
#   function: proj_h_mat             #
######################################

def proj_h_mat(my_L, my_R, my_W):
    """Returns the projected h matrix

    Parameters
    ----------
    my_L : 
    my_R : 
    my_W : 

    Returns
    -------
    res : the projected matrix

    """
    res = np.tensordot(my_W, my_R, axes=(2,0))
    res = np.tensordot(my_W, res, axes=(2,0))
    res = np.tensordot(my_L, res, axes=(1,0))
    # transpose in order to be the right in vector
    res = np.transpose(res,(0,2,4,6,1,3,5,7))

    return res

######################################
#   function: proj_h_vec_mul         #
######################################

def proj_h_vec_mul(x, my_L, my_R, my_W):
    """Computes the action of the projected Hamiltonian LWWR onto
    the vector x.

    Parameters
    ----------
    x : It is supposed to have shape d^2 D^2 and reshaped as
        D,d,d,D has first D as left index, first d as 2nd, etc.
        as coming from compute_psi_guess
    my_L : 
    my_R : 
    my_W : 

    Returns
    -------
    res : the multiplied vector

    """

    D = my_L.shape[0]
    d = my_W.shape[3]
    # reshape x
    res = x.reshape((D,d,d,D))
    # act with R
    res = np.tensordot(res, my_R, axes=(3,2))
    # with W
    res = np.tensordot(my_W, res, axes=([3,2],[2,3]))
    # with W
    res = np.tensordot(my_W, res, axes=([3,2],[3,0]))
    # with L
    res = np.tensordot(my_L, res, axes=([1,2],[0,3]))
    # reshape
    res = res.reshape(D*D*d*d)

    return res

######################################
#   function: truncate_and_update    #
######################################

def truncate_and_update(my_psi, cur_D, my_d, trunc_D):
    """Computes the SVD of my_psi, which (after reshaping maybe) has shape
    (cur_D,d,d,cur_D). Then updates A,Lambda,B and returns truncation
    error.

    Parameters
    ----------
    my_psi : wave function for two spins, dim=cur_D*cur_D*d*d
    cur_D : current aux dim
    my_d : phys dim
    trunc_D : maximal auxiliary dim

    Returns
    -------
    In this order
    A :
    Lambda :
    B :
    err :

    """

    # Perform an SVD
    psi = my_psi.reshape((cur_D*my_d,my_d*cur_D))
    # SVD: if psi.shape == (d D_L, d D_R), then when full_matrices=False 
    # it returns U.shape = (d D_L, new_D), Vh.shape = (new_D, d D_R), with 
    # new_D = min(d D_L, d D_R). Here D_L = D_R = cur_D, so new_D = cur_D*d
    # linalg.svd returns in basis with descending order: s[1]>s[2]> ...
    U, s, Vh = linalg.svd(psi, full_matrices=False)

    # In some cases, eg itebd, psi is not normalized: sum s^2 \neq 1.
    # then makes sure it is, since we assume it when truncating
    s_norm = np.sqrt(sum(s**2))
    if np.allclose(s_norm, 1.) == False:
        s = s/s_norm

    small_num = 10.**(-12)
    # Truncate: np.sum(s>small_num), returns the number of elements
    # of the array s whose value is > small_num, where all the weight
    # of the singular value is concentrated...it is the effective
    # value of new_D. Truncate to trunc_D = maximum value of D given
    # from input, if it gets too large.
    new_D = np.min([np.sum(s>small_num), trunc_D])
    # So keep only the new_D elements in second index of U and assign it to A:
    A = (U[:,:new_D]).reshape((cur_D,my_d,new_D))
    # Assign first new_D elements of s to Lambda and divide by their
    # sum squared, which is the norm of the wave function at this
    # stage, since A,B normalized.
    Lambda = s[:new_D]/np.sqrt(np.sum(s[:new_D]**2))
    B = (Vh[:new_D,:]).reshape((new_D,my_d,cur_D))

    # Truncation error = 1 - sum_{a=1}^D s_a
    err = 1. - sum(s[:new_D]**2)

    return A, Lambda, B, err

######################################
#       function: check_norm         #
######################################

def check_norm(my_A, my_B, my_Lambda):
    """Check normalization of A,B,Lambda guaranteeing the normalization 
    of the wave function itself.

    Parameters
    ----------
    my_A :
    my_B :
    my_Lambda :

    Returns
    -------
    True if normalized, 1 False not normalized

    """
    is_norm = True
    D = my_A.shape[2]
    # Check A
    Astar=np.conjugate(my_A)
    A_norm_mat = np.tensordot(my_A,Astar,axes=([0,1],[0,1]))
    #print A_norm_mat
    if np.allclose(A_norm_mat , np.eye(D)) == False:
        is_norm = False
        print ("A not normalized", A_norm_mat)
    # Check B
    Bstar=np.conjugate(my_B)
    B_norm_mat = np.tensordot(my_B,Bstar,axes=([1,2],[1,2]))
    #print B_norm_mat
    if np.allclose(B_norm_mat , np.eye(D)) == False:
        is_norm = False
        print ("B not normalized", B_norm_mat)
    # Check Lambda
    Lambda_norm = linalg.norm(my_Lambda)
    #print Lambda_norm
    if np.allclose( Lambda_norm , 1.) == False:
        is_norm = False
        print ("Lambda not normalized", Lambda_norm)

    return is_norm

######################################
#  function: check_norm_transfer_op  #
######################################

def check_norm_transfer_op(my_A, my_Lambda, my_B, my_Lambda_old, my_atol):
    """Check normalization of the left/right transfer matrices associated
    to the unit cells A.Lambda.B.Lambda_old^(-1) and Lambda_old^(-1).A.Lambda.B

    Parameters
    ----------
    my_A :
    my_B :
    my_Lambda :
    my_Lambda_old :

    Returns
    -------
    True if normalized, 1 False not normalized

    """
    is_norm = True
    d = my_A.shape[1]
    E_left = transfer_operator(np.tensordot(my_A,np.diag(my_Lambda),\
                                            axes=(2,0)), \
                               np.tensordot(my_B,linalg.inv(my_Lambda_old),\
                                            axes=(2,0)), \
                               np.eye(d),np.eye(d))
    DD=E_left.shape[0]
    norm= np.tensordot(E_left,np.eye(DD),axes=([0,1],[0,1]))
    if np.allclose(norm , np.eye(DD), atol=my_atol) == False:
        print norm
        print ("Error: not left normalized")
        is_norm = False
    E_right = transfer_operator(np.tensordot(linalg.inv(my_Lambda_old),\
                                             my_A, axes=(1,0)), \
                                np.tensordot(np.diag(my_Lambda),my_B,\
                                             axes=(1,0)), \
                                np.eye(d),np.eye(d))
    DD=E_right.shape[2]
    norm= np.tensordot(E_right,np.eye(DD),axes=([2,3],[0,1]))
    if np.allclose(norm , np.eye(DD), atol=my_atol) == False:
        print norm
        print ("Error: not right normalized")
        is_norm = False
        
    return is_norm

######################################
#   function: rotate_center          #
######################################

def rotate_center(my_A, my_Lambda, my_B):
    """Rotates the "center" in McCulloch terminology, once to left and
    once to the right. Namely, it computes SVD of Lambda.B to A_new,
    Lambda_R, and A.Lambda to Lambda_L, B_new

    Parameters
    ----------
    my_A :
    my_Lambda :
    my_B :

    Returns
    -------
    In this order
    A_tilde :
    Lambda_R :
    Lambda_L :
    B_tilde :

    """

    # TODO: parallelize

    A_tilde, Lambda_R = rotate_right(my_Lambda, my_B)
    Lambda_L, B_tilde = rotate_left(my_A, my_Lambda)
    
    return A_tilde, Lambda_R, Lambda_L, B_tilde

######################################
#   function: rotate_right           #
######################################

def rotate_right(my_Lambda, my_B):
    """Rotates the "center" in McCulloch terminologu, once to the right.

    Parameters
    ----------
    my_Lambda :
    my_B :

    Returns
    -------
    In this order
    A_tilde :
    Lambda_R :

    """
    D, d, Dp = my_B.shape
    mat = np.tensordot(np.diag(my_Lambda),my_B, axes=(1,0))
    mat = mat.reshape(D*d,Dp)
    U, s, Vh = linalg.svd(mat, full_matrices=False)
    A_tilde = U.reshape(D,d,U.shape[1])
    Lambda_R = np.dot(np.diag(s),Vh)

    return A_tilde, Lambda_R

######################################
#   function: rotate_left            #
######################################

def rotate_left(my_A, my_Lambda):
    """Rotates the "center" in McCulloch terminologu, to the left

    Parameters
    ----------
    my_A :
    my_Lambda :

    Returns
    -------
    In this order
    Lambda_L :
    B_tilde :

    """
    
    Dp, d, D = my_A.shape
    mat = np.tensordot(my_A, np.diag(my_Lambda), axes=(2,0))
    mat = mat.reshape(Dp,D*d)
    U, s, Vh = linalg.svd(mat, full_matrices=False)
    B_tilde = Vh.reshape(Vh.shape[0],d,D)
    Lambda_L = np.dot(U,np.diag(s))

    return Lambda_L, B_tilde

######################################
#   function: f_ortho                #
######################################

def f_ortho(Lambda_1, Lambda_2):
    """Computes the fidelity F(rho_1,rho_2) of the density matrices rho_1
    = Lambda_1^\dagger Lambda_1 and rho_2 = Lambda_2^\dagger
    Lambda_2. Namely: sum of singular values of np.dot(Lambda_1,
    np.conj(Lambda_2).T) Returns F

    Parameters
    ----------
    Lambda_1 : 
    Lambda_2 :
    accuracy :

    Returns
    -------
    F :

    """

    # Creates matrix with the product of Lambda_1*Lambda_2^\dagger
    # and computes its singular value. This correspond to the fidelity.
    # First check if square matrices:
    shape_1 = Lambda_1.shape
    shape_2 = Lambda_2.shape
    
    # Check if parameters are matrices, necessary
    if len(shape_1) != 2 or len(shape_2) != 2 or \
       shape_1[0] != shape_1[1] or shape_2[0] != shape_2[1]:
        print ("Error in f_ortho: not square matrices...")
        print (shape_1, shape_2)
        return 0

    # Check if parameters are same size, atteined only after some
    # transient, where the dimension grows. It is necessary for convergence but
    # not an error if not, it just means that more iterations are needed.
    if shape_1[0] != shape_2[0]:
        # do nothing, just go on and wait
        # print "In converged: different shapes ", shape_1[0], shape_2[0]
        fidelity = 0
    else:
        mat = np.dot(Lambda_1, np.conj(Lambda_2).T)
        U, s, Vh = linalg.svd(mat, full_matrices=False)
        fidelity = np.sum(s)

    return fidelity

######################################
#    function: compute_X_Y_eta       #
######################################

def compute_eta_X_Y(my_A, my_Lambda, my_B, my_Lambda_old, mytype):
    """Use gauge dof to rotate A,B,Lambda_old in such a way that the state
    is normalized, aka that the identity is an eigenvector of left and
    right transfer matrices. Returns the new Lambda,X,Y and the max eig eta

    Parameters
    ----------
    my_A :
    my_Lambda : 
    my_B :
    my_Lambda_old : 
    mytype :

    Returns
    -------
    max_eig_L : 
    X :
    Y :

    """
    
    # Compute the gauge factors X,Y and the max eigenvalues of
    # left/right transfer operators (in practice 1 but take care of
    # general case)
    X, max_eig_L = compute_X(my_A, my_Lambda, my_B, my_Lambda_old, mytype)
    Y, max_eig_R = compute_Y(my_A, my_Lambda, my_B, my_Lambda_old, mytype)

    if np.allclose(max_eig_L,max_eig_R)==False:
        print ("ERROR: In orthogonalize_unit_cell: max_eig_L!=max_eig_R")
        return 0,0,0,0

    return  max_eig_L, X, Y


######################################
# function: left_normalize_unit_cell #
######################################

def left_normalize_unit_cell(my_A, my_Lambda, my_B, my_Lambda_old):
    """Computes from an SVD, the left normalized matrices A1,A2.
    Then normalize the matrix my_Lambda_old by the sum of its singular values
    squared.

    Parameters
    ----------
    my_A :
    my_Lambda : 
    my_B :
    my_Lambda_old : 

    Returns
    -------
    A1 :
    A2 :
    Lambda_old_normalized

    """
    
    # Computes the matrix representing the unit cell wave function psi_cell
    psi_cell = np.tensordot(my_A, np.diag(my_Lambda), axes=(2,0))
    psi_cell = np.tensordot(psi_cell, my_B, axes=(2,0))
    # In this basis and at this point my_Lambda_old is in general not
    # diagonal anymore.
    if my_Lambda_old.ndim == 1:
        psi_cell = np.tensordot(psi_cell, np.diag(1/my_Lambda_old), axes=(3,0))
    elif my_Lambda_old.ndim == 2:
        psi_cell = np.tensordot(psi_cell, linalg.inv(my_Lambda_old), axes=(3,0))
    else:
        print ("ERROR: in left_normalize_unit_cell, my_Lambda_old has wrong ndim")

    # here D',d,d,D'
    Dp = psi_cell.shape[0]; d = psi_cell.shape[1]; Dpd = Dp*d
    psi_cell = psi_cell.reshape(Dpd,Dpd)
    # Do an SVD: U,Vh is Dp*d x Dp*d, while s is Dp*d
    U, s, Vh = linalg.svd(psi_cell, full_matrices=False)
    A1 = U.reshape((Dp,d,Dpd))
    A2 = (np.dot(np.diag(s),Vh)).reshape((Dpd, d, Dp))

    # Normalize Lambda_old by the sum squared of its singular values
    if my_Lambda_old.ndim == 1:
        n = linalg.norm(my_Lambda_old)
        Lambda_old_normalized = my_Lambda_old/n
    elif my_Lambda_old.ndim == 2:
        U, s, Vh = linalg.svd(my_Lambda_old, full_matrices=False)
        Lambda_old_normalized = my_Lambda_old/np.sqrt(np.dot(s,np.conj(s)))
    else:
        print ("ERROR: in left_normalize_unit_cell, my_Lambda_old has wrong ndim")

    print ("Exiting left_normalize_unit_cell")
    
    return  A1, A2, Lambda_old_normalized

######################################
# function: mixed_repr_unit_cell #
######################################

def mixed_repr_unit_cell(my_A, my_Lambda, my_B, my_Lambda_old, X, Y):
    """Computes from an SVD, the left normalized matrices A1,A2 and the
    right normalized B1,B2.  Then normalize the matrix my_Lambda_old
    by the sum of its singular values squared. This guarantees wf is
    normalized.

    Parameters
    ----------
    my_A :
    my_Lambda : 
    my_B :
    my_Lambda_old : 
    X :
    Y :

    Returns
    -------
    A1 :
    A2 :
    Lambda_old_normalized
    B1 :
    B2 :

    """
    
    # Computes the matrix representing the unit cell wave function psi_cell
    AL = np.tensordot(my_A, np.diag(my_Lambda), axes=(2,0))
    ALB = np.tensordot(AL, my_B, axes=(2,0))

    # First AB unit cell. 
    XLold = np.dot(X, np.diag(my_Lambda_old))
    psi_cell = np.tensordot(X, np.tensordot(ALB, linalg.inv(XLold),\
                                            axes=(3,0)), axes=(1,0))

    # here D',d,d,D'
    Dp = psi_cell.shape[0]; d = psi_cell.shape[1]; Dpd = Dp*d
    # check normalization
    idmat1 = np.tensordot(np.conj(psi_cell),psi_cell,axes=([0,1,2],[0,1,2]))
    if np.allclose(idmat1, np.eye(Dp)) == False:
        print ("ERROR: not idmat X")


    psi_cell = psi_cell.reshape(Dpd,Dpd)
    # Do an SVD: U,Vh is Dp*d x Dp*d, while s is Dp*d
    U, s, Vh = linalg.svd(psi_cell, full_matrices=False)
    A1 = U.reshape((Dp,d,Dpd))
    A2 = (np.dot(np.diag(s),Vh)).reshape((Dpd, d, Dp))

    # Now BA unit cell. 
    LoldY = np.dot(np.diag(my_Lambda_old), Y)
    psi_cell = np.tensordot(np.tensordot(linalg.inv(LoldY),ALB,\
                                         axes=(1,0)), Y, axes=(3,0))
    # here D',d,d,D'
    Dp = psi_cell.shape[0]; d = psi_cell.shape[1]; Dpd = Dp*d

    # check normalization
    idmat1 = np.tensordot(np.conj(psi_cell),psi_cell,axes=([1,2,3],[1,2,3]))
    if np.allclose(idmat1, np.eye(Dp)) == False:
        print ("ERROR: not idmat Y")

    psi_cell = psi_cell.reshape(Dpd,Dpd)
    # Do an SVD: U,Vh is Dp*d x Dp*d, while s is Dp*d
    U, s, Vh = linalg.svd(psi_cell, full_matrices=False)
    B2 = Vh.reshape(Dpd,d,Dp)
    B1 = (np.dot(U,np.diag(s))).reshape((Dp, d, Dpd))

    # Check A1
    A = A1
    D = A.shape[2]
    A_norm_mat = np.tensordot(A,np.conj(A),axes=([0,1],[0,1]))
    if np.allclose(A_norm_mat , np.eye(D)) == False:
        print "Error: A1 not normalized", A_norm_mat
    # Check A2
    A = A2
    D = A.shape[2]
    A_norm_mat = np.tensordot(A,np.conj(A),axes=([0,1],[0,1]))
    if np.allclose(A_norm_mat , np.eye(D), atol=1e-08) == False:
        print "Error: A2 not normalized", A_norm_mat 
    # Check B1
    B = B1
    D = B.shape[0]
    B_norm_mat = np.tensordot(B,np.conj(B),axes=([1,2],[1,2]))
    if np.allclose(B_norm_mat , np.eye(D)) == False:
        print "Error: B1 not normalized", B_norm_mat
    # Check B2
    B = B2
    D = B.shape[0]
    B_norm_mat = np.tensordot(B,np.conj(B),axes=([1,2],[1,2]))
    if np.allclose(B_norm_mat , np.eye(D)) == False:
        print "Error: B2 not normalized", B_norm_mat
 
    # Multiply Lambda_old by X,Y and divide by the sum squared of its
    # singular values. This ensures that the wave function has norm = 1
    XLoldY = np.dot(XLold, Y)
    U, s, Vh = linalg.svd(XLoldY, full_matrices=False)
    n = np.sqrt(np.dot(s,np.conj(s)))

    new_Lold_normalized = XLoldY/n

    return  A1, A2, new_Lold_normalized, B1, B2

######################################
#       function: compute_X          #
######################################

def compute_X(my_A, my_Lambda, my_B, my_Lambda_old, mytype):
    """Computes X by diagoanalizing the left transfer matrix of the unit
    cell and compute the dominant eigenvector V_L, V_L = X^\dagger X

    Parameters
    ----------
    my_A :
    my_Lambda : 
    my_B :
    my_Lambda_old : 
    mytype :

    Returns
    -------
    X :
    max_eig :

    TODO: take care of non invertible lambda_old

    """
    
    # Compute A_tilde, Lambda_R, P
    A_tilde, Lambda_R = rotate_right(my_Lambda, my_B)
    if my_Lambda_old.ndim == 1:
        P = np.dot(Lambda_R, np.diag(1/my_Lambda_old))
    elif my_Lambda_old.ndim == 2:
        P = np.dot(Lambda_R, linalg.inv(my_Lambda_old))
    else:
        print ("ERROR: in compute_X, my_Lambda_old has wrong ndim")

    mv = lambda vec: matvec_E_L(vec, my_A, A_tilde, P)
    # Dimensions and check if the transfer operator E_L is square in order 
    # to be able to talk about eigenvectors and eigenvalues.
    D = my_A.shape[0];
    if D != P.shape[1]:
        print ("ERROR: In compute_X, P.shape[1] ",  P.shape[1], \
            " != my_A.shape[0]", D)
        return 0
    # Otherwise continue and create linear operator od dim D^2 x D^2
    E_L = LinearOperator((D*D, D*D), matvec=mv, dtype=mytype)    
    # Compute the dominant eigenvector of E_L, no initial state
    # provided. Note, the transfer operator is Hermitian in general, so
    # remove the h... It proves that with k=1 it does not converge.
    if D <= 2:
        vals,vecs = eigs(E_L, k=2, which='LM', return_eigenvectors=True)
    else:
        vals,vecs = eigs(E_L, k=3, which='LM', return_eigenvectors=True)
    max_ind = np.where(vals==max(vals))[0][0]
    max_eig = vals[max_ind]
    # This trick avoids that the eigenvalue has an overall phase which spoils
    # then hermiticity...
    V_L = vecs[:,max_ind]/vecs[:,max_ind][0]
    V_L = V_L/linalg.norm(V_L)
    V_L = V_L.reshape(D,D)
    # Check:
    if np.allclose(E_L.matvec(vecs[:,max_ind]),\
                   max_eig*vecs[:,max_ind]) == False:
        print ("ERROR: In compute_X: eig with low accuracy ")

    # Why V_L hermitian and >=0? TODO
    V_L_dag = np.transpose(np.conj(V_L))
    if np.allclose(V_L, V_L_dag) == False:
        print ("ERROR: first In compute_X: V_L not hermitian. ")
        return 0

    if np.allclose(V_L, V_L_dag) == False:
        print ("ERROR: In compute_X: V_L not hermitian ")
        return 0
    w, v = linalg.eigh(V_L)
    # and if V_L non-negative - as it should be. We have seen that
    # sometimes the global sign is wrong.  Clearly this can be
    # reabsorbed in the definition of V_L, so check if all negative,
    # just return -w
    if np.all(w<=0):
        sign = -1.
    elif np.all(w>=0): 
        sign = 1.
    else: #means neither all >=0 nor all <=0, sign changes.
        print ("ERROR: In compute_X: V_L not >= 0 ")
        print (w)
        return 0    
    # Finally compute V_L = X^dagger X:
    w = sign*w
    X = np.dot(np.diag(np.sqrt(w)), np.transpose(np.conj(v)))
    # Sanity check
    if np.allclose(np.dot(np.transpose(np.conj(X)),X), sign*V_L) == False:
        print ("ERROR: In compute_X: X with low accuracy ")

    if np.allclose(E_L.matvec(np.dot(np.transpose(np.conj(X)),X).reshape(D*D)),\
                   max_eig*np.dot(np.transpose(np.conj(X)),X).reshape(D*D)) \
        == False:
        print ("ERROR: In compute_X: eig with low accuracy 2 ")

    # Check that for the rotated transfer operator, E(1)=1:
    # new_A = np.tensordot(X, my_A, axes=(1,0))
    # new_P = np.dot(P, linalg.inv(X))/np.sqrt(max_eig) 
    # vec_id = np.eye(D).reshape(D*D)
    # print (np.allclose(matvec_E_L(vec_id, new_A, A_tilde, new_P),\
    #                   vec_id))

    return X, max_eig


######################################
#       function: compute_X_2        #
######################################

def compute_X_2(my_A, my_Lambda, my_B, my_Lambda_old):
    """Alternative, just for testing.

    Parameters
    ----------
    my_A :
    my_Lambda : 
    my_B :
    my_Lambda_old : 

    Returns
    -------
    X :

    """
    X=0
    
    # Compute A_tilde, Lambda_R, P
    A_tilde, Lambda_R = rotate_right(my_Lambda, my_B)
    P = np.dot(Lambda_R, np.diag(1/my_Lambda_old))

    # Construct the matrix explicitly:
    D = my_A.shape[0]; d = my_A.shape[1]
    M = np.tensordot(A_tilde, P, axes=(2,0))
    M = np.tensordot(my_A, M, axes=(2,0))
    print (M.shape)
    E_L_mat = np.zeros((D*D,D*D))
    for s1 in range(d):
        for s2 in range(d):
            E_L_mat += np.kron(np.conj(M[:,s1,s2,:]),M[:,s1,s2,:])        

    print (E_L_mat.shape)

    myid=np.eye(D).reshape(D*D)
    EL_id = np.dot(myid ,E_L_mat).reshape(D,D)
    print (np.allclose(EL_id , np.dot(np.transpose(np.conj(P)),P)))

    vals,vecs = eigs(E_L_mat, k=5, which='LM', return_eigenvectors=True)
    V_L = vecs[:,0].reshape(D,D)
    V_L_dag = np.transpose(np.conj(V_L))

    return X


######################################
#       function: compute_Y          #
######################################

def compute_Y(my_A, my_Lambda, my_B, my_Lambda_old, mytype):
    """Computes Y by diagoanalizing the right transfer matrix of the unit
    cell and compute the dominant eigenvector V_R, V_R = YY^dagger
    Note: shifted unit cell wrt compute X.

    Parameters
    ----------
    my_A :
    my_Lambda : 
    my_B :
    my_Lambda_old : 
    mytype :

    Returns
    -------
    Y :
    max_eig :

    """
    
    # Compute Lambda_L, B_tilde, Q
    Lambda_L, B_tilde = rotate_left(my_A, my_Lambda)
    if my_Lambda_old.ndim == 1:
        Q = np.dot(np.diag(1/my_Lambda_old), Lambda_L)
    elif my_Lambda_old.ndim == 2:
        Q = np.dot(linalg.inv(my_Lambda_old), Lambda_L)
    else:
        print ("ERROR: in compute_Y, my_Lambda_old has wrong ndim")

    mv = lambda vec: matvec_E_R(vec, Q, B_tilde, my_B)
    # Dimensions and check if the transfer operator E_R is square in order 
    # to be able to talk about eigenvectors and eigenvalues.
    D = my_B.shape[2];
    if D != Q.shape[0]:
        print ("ERROR: In compute_Y, Q.shape[0] ",  Q.shape[0], \
            " != my_B.shape[2]", D)
        return 0
    # Otherwise continue and create linear operator of dim D^2 x D^2
    E_R = LinearOperator((D*D, D*D), matvec=mv, dtype=mytype)

    # Check if the identity goes to QQ^dagger
    # myid=np.eye(D).reshape(D*D)
    # ER_id = E_R.matvec(myid).reshape(D,D)
    # print "check id in compute_Y", np.allclose(ER_id , np.dot(Q,np.transpose(np.conj(Q))))

    # Compute the dominant eigenvector of E_R, no initial state provided
    # See comments in compute_X
    if D <= 2:
        vals,vecs = eigs(E_R, k=2, which='LM', return_eigenvectors=True)
    else:
        vals,vecs = eigs(E_R, k=3, which='LM', return_eigenvectors=True)
    # Why V_R hermitian and >=0? TODO
    max_ind = np.where(vals==max(vals))[0][0]
    max_eig = vals[max_ind]
    # This trick avoids that the eigenvalue has an overall phase which spoils
    # then hermiticity...
    V_R = vecs[:,max_ind]/vecs[:,max_ind][0]
    V_R = V_R/linalg.norm(V_R)
    V_R = V_R.reshape(D,D)
    V_R_dag = np.transpose(np.conj(V_R))
    if np.allclose(V_R, V_R_dag) == False:
        print ("ERROR: In compute_Y: V_R not hermitian. ")
        return 0
    w, v = linalg.eigh(V_R)
    # Check sign of w as in compute_X
    if np.all(w<=0):
        sign = -1.
    elif np.all(w>=0): 
        sign = 1.
    else: #means neither all >=0 nor all <=0, sign changes.
        print ("ERROR: In compute_Y: V_R not >= 0 ")
        print (w)
        return 0    
    # Finally compute V_R = YY^dagger :
    w = sign*w
    Y = np.dot(v, np.diag(np.sqrt(w)))
    # Sanity check
    if np.allclose(np.dot(Y,np.transpose(np.conj(Y))), sign*V_R) == False:
        print ("ERROR: In compute_Y: Y with low accuracy ")

    if np.allclose(E_R.matvec(np.dot(Y,np.transpose(np.conj(Y))).reshape(D*D)),\
                   max_eig*np.dot(Y,np.transpose(np.conj(Y))).reshape(D*D)) \
        == False:
        print ("ERROR: In compute_Y: eig with low accuracy 2 ")


    return Y, max_eig

######################################
#       function: matvec_E_L         #
######################################

def matvec_E_L(vec, A, A_tilde, P):
    """Computes the action of the left transfer matrix associated to the
    unit cell represented by the matrices A.A_tilde.P on the vector
    vec from the left.

    Parameters
    ----------
    vec : the input vector to be multiplied, dim = D^2
    A :
    A_tilde : 
    P : 

    Returns
    -------
    ret : the multiplied vector

    """

    # Reshape
    Dp = A.shape[0]
    ret = vec.reshape(Dp,Dp)

    tmp = np.tensordot(A, A_tilde, axes=(2,0))
    tmp = np.tensordot(tmp, P, axes=(3,0)) # AAP
    tmp = np.tensordot(np.conj(tmp), tmp, axes=([1,2],[1,2])) # E_L
    ret = np.tensordot(tmp, ret, axes = ([0,2],[0,1]))

    # Reshape
    ret = ret.reshape(Dp*Dp)

    return ret

######################################
#       function: matvec_E_R         #
######################################

def matvec_E_R(vec, Q, B_tilde, B):
    """Computes the action of the right transfer matrix associated to the
    unit cell represented by the matrices Q.B_tilde.B on the vector
    vec from the right.

    Parameters
    ----------
    vec : the input vector to be multiplied, dim = D^2
    Q :
    B_tilde : 
    B : 

    Returns
    -------
    ret : the multiplied vector

    """

    # Reshape
    Dp = B.shape[2]
    ret = vec.reshape(Dp,Dp)

    tmp = np.tensordot(Q, B_tilde, axes=(1,0))
    tmp = np.tensordot(tmp, B, axes=(2,0)) # QBB
    tmp = np.tensordot(np.conj(tmp), tmp, axes=([1,2],[1,2])) # E_R
    ret = np.tensordot(tmp, ret, axes = ([3,1],[0,1]))
    # Need to transpose
    ret = np.transpose(ret)

    # Reshape
    ret = ret.reshape(Dp*Dp)

    return ret

######################################
#  function: two_pf_single_site_op   #
######################################

def two_pf_single_site_op(my_A1, my_A2, my_Lambda_old, Op_1, Op_i, i):
    """Computes the correlator <psi|Op_1(1)Op_i(i)|psi>.
    Assumes <psi|psi>=1

    Parameters
    ----------
    my_A1, my_A2 : matrices representing the unit cell
    my_Lambda_old :
    Op_1 : Assume at site 1 since translation invariance
    Op_i : 
    i : 

    Returns
    -------
    corr : the desired correlator

    """

    # Check shapes
    d = my_A1.shape[1]
    if Op_1.shape != (d,d) or Op_i.shape != (d,d):
        print ("ERROR: in two_pf_single_site_op: wrong shapes of Ops")
        return 0
    
    # Check parity of i
    if np.mod(i, 2) == 0:
        c_i = i/2 # cell where site i is
        E_Op_i = transfer_operator(my_A1, my_A2, np.eye(d), Op_i)
    else:
        c_i = (i+1)/2 # cell where site i is
        E_Op_i = transfer_operator(my_A1, my_A2, Op_i, np.eye(d))

    # Compute explicitly the transfer operators of the unit cell
    E_id = transfer_operator(my_A1, my_A2)
    E_Op_1 = transfer_operator(my_A1, my_A2, Op_1, np.eye(d))

    # Recursion to compute C
    D = my_A1.shape[0]
    C = np.eye(D)
    for k in range(1,c_i+1): #k=1,...,c_i
        if k == 1:
            C = act_transfer_operator_left(E_Op_1, C)
        elif k == c_i:
            C = act_transfer_operator_left(E_Op_i, C)
        else:
            C = act_transfer_operator_left(E_id, C)

    # Correlator:
    corr = np.tensordot(C, my_Lambda_old, axes=(1,0))
    corr = np.tensordot(corr, np.conj(my_Lambda_old), axes=([0,1],[0,1]))

    return corr

######################################
#  function: two_pf_bond_op          #
######################################

def two_pf_bond_op(my_A1, my_A2, my_Lambda_old, Op_1, Op_i, i):
    """Computes the correlator <psi|Op_1(1)Op_i(i)|psi> where Op_1
    is a bond operator, shape (d^2,d^2). Assumes <psi|psi>=1

    Parameters
    ----------
    my_A1, my_A2 : matrices representing the unit cell
    my_Lambda_old :
    Op_1 : Assume at site 1 since translation invariance
    Op_i : 
    i : 

    Returns
    -------
    corr : the desired correlator

    """

    # Check shapes
    d = my_A1.shape[1]
    if Op_1.shape != (d**2,d**2) or Op_i.shape != (d**2,d**2):
        print ("ERROR: in two_pf_bond_op: wrong shapes of Ops")
        return 0
    
    # Check parity of i
    if np.mod(i, 2) == 0:
        c_i = i/2
        # recast the operator into a four sites one and use the
        # transfer_operator routine
        four_sites_eff_op = np.kron(np.eye(d),Op_i)
        four_sites_eff_op = np.kron(four_sites_eff_op,np.eye(d))
        E_Op_i = transfer_operator(my_A1, my_A2, four_sites_eff_op)
    else:
        c_i = (i+1)/2 # cell where site i is
        E_Op_i = transfer_operator(my_A1, my_A2, Op_i)

    # Compute explicitly the transfer operators of the unit cell
    E_id = transfer_operator(my_A1, my_A2)
    E_Op_1 = transfer_operator(my_A1, my_A2, Op_1)

    # Recursion to compute C
    D = my_A1.shape[0]
    C = np.eye(D)
    for k in range(1,c_i+1): #k=1,...,c_i
        if k == 1:
            C = act_transfer_operator_left(E_Op_1, C)
        elif k == c_i:
            C = act_transfer_operator_left(E_Op_i, C)
        else:
            C = act_transfer_operator_left(E_id, C)

    # Correlator:
    corr = np.tensordot(C, my_Lambda_old, axes=(1,0))
    corr = np.tensordot(corr, np.conj(my_Lambda_old), axes=([0,1],[0,1]))

    return corr

######################################
#  function: two_pf_4_sites_op       #
######################################

def two_pf_4_sites_op(my_A1, my_A2, my_Lambda_old, Op_1, Op_i, i):
    """Computes the correlator <psi|Op_1(1)Op_i(i)|psi> where Op_1
    is a 4 sites operator, shape (d^4,d^4). Assumes <psi|psi>=1

    Parameters
    ----------
    my_A1, my_A2 : matrices representing the unit cell
    my_Lambda_old :
    Op_1 : Assume at site 1 since translation invariance
    Op_i : 
    i : 

    Returns
    -------
    corr : the desired correlator

    """

    # Check shapes
    d = my_A1.shape[1]
    if Op_1.shape != (d**4,d**4) or Op_i.shape != (d**4,d**4):
        print ("ERROR: in two_pf_4_sites_op: wrong shapes of Ops")
        return 0
    
    # Check parity of i
    if np.mod(i, 2) == 0:
        c_i = i/2 
        # Create an effective six sites operator and use transfer
        # operator routine
        six_sites_eff_op = np.kron(np.eye(d), Op_i)
        six_sites_eff_op = np.kron(six_sites_eff_op, np.eye(d))
        E_Op_i = transfer_operator(my_A1, my_A2, six_sites_eff_op)
    else:
        c_i = (i+1)/2 # cell where site i is
        E_Op_i = transfer_operator(my_A1, my_A2, Op_i)

    # Compute explicitly the transfer operators of the unit cell
    E_id = transfer_operator(my_A1, my_A2)
    E_Op_1 = transfer_operator(my_A1, my_A2, Op_1)

    # Recursion to compute C
    D = my_A1.shape[0]
    C = np.eye(D)
    for k in range(1,c_i+1): #k=1,...,c_i
        if k == 1:
            C = act_transfer_operator_left(E_Op_1, C)
        elif k==2:
            continue # E_op_1 is 4 sites
        elif k == c_i:
            C = act_transfer_operator_left(E_Op_i, C)
        else:
            C = act_transfer_operator_left(E_id, C)

    # Correlator:
    corr = np.tensordot(C, my_Lambda_old, axes=(1,0))
    corr = np.tensordot(corr, np.conj(my_Lambda_old), axes=([0,1],[0,1]))

    return corr

######################################
#  function: two_pf_6_sites_op       #
######################################

def two_pf_6_sites_op(my_A1, my_A2, my_Lambda_old, Op_1, Op_i, i):
    """Computes the correlator <psi|Op_1(1)Op_i(i)|psi> where Op_1, Op_i
    is a 6 sites operator, shape (d^6,d^6). Assumes <psi|psi>=1

    Parameters
    ----------
    my_A1, my_A2 : matrices representing the unit cell
    my_Lambda_old :
    Op_1 : Assume at site 1 since translation invariance
    Op_i : 
    i : 

    Returns
    -------
    corr : the desired correlator

    """

    # Check shapes
    d = my_A1.shape[1]
    if Op_1.shape != (d**6,d**6) or Op_i.shape != (d**6,d**6):
        print ("ERROR: in two_pf_6_sites_op: wrong shapes of Ops")
        return 0
    
    # Check parity of i
    if np.mod(i, 2) == 0:
        c_i = i/2 
        # Create an effective 8 sites operator and use transfer
        # operator routine
        eight_sites_eff_op = np.kron(np.eye(d), Op_i)
        eight_sites_eff_op = np.kron(eight_sites_eff_op, np.eye(d))
        E_Op_i = transfer_operator(my_A1, my_A2, eight_sites_eff_op)
    else:
        c_i = (i+1)/2 # cell where site i is
        E_Op_i = transfer_operator(my_A1, my_A2, Op_i)

    # Compute explicitly the transfer operators of the unit cell
    E_id = transfer_operator(my_A1, my_A2)
    E_Op_1 = transfer_operator(my_A1, my_A2, Op_1)

    # Recursion to compute C
    D = my_A1.shape[0]
    C = np.eye(D)
    for k in range(1,c_i+1): #k=1,...,c_i
        if k == 1:
            C = act_transfer_operator_left(E_Op_1, C)
        elif k==2:
            continue # E_op_1 is 4 sites
        elif k == c_i:
            C = act_transfer_operator_left(E_Op_i, C)
        else:
            C = act_transfer_operator_left(E_id, C)

    # Correlator:
    corr = np.tensordot(C, my_Lambda_old, axes=(1,0))
    corr = np.tensordot(corr, np.conj(my_Lambda_old), axes=([0,1],[0,1]))

    return corr

######################################
#    function: transfer_operator     #
######################################

def transfer_operator(my_A1, my_A2, *Ops):
    """Returns the transfer operator associated to the unit cell A1,A2
    with inserted operators *Ops = variable number of arguments
    between 0 and 2. 0 Means just to put the identity, 1 that a bond
    operator, 4 sites, or 6 operator is given, and 2 that Op1 and Op2
    are given.

    Parameters
    ----------
    my_A1 :
    my_A2 :
    *Ops :

    Returns
    -------
    ret : the transfer operator, 0------2
                                   |  |
                                   O  O
                                   |  |
                                 1------3 , 
          where the numbers stand for the position of the indices

    """
    
    n_args = len(Ops)
    AA = np.tensordot(my_A1, my_A2, axes=(2,0)) 
    if n_args == 0:
        ret = np.tensordot(np.conj(AA), AA, axes=([1,2],[1,2]))
    elif n_args == 1:
        d = my_A1.shape[1]
        Op_size = Ops[0].size
        if Op_size == d**4: #bond operator
            my_op = Ops[0].reshape(d,d,d,d)
            # Contraction
            ret = np.tensordot(my_op, AA, axes=([2,3],[1,2]))
            ret = np.tensordot(np.conj(AA), ret, axes=([1,2],[0,1]))
        elif Op_size == d**8: #4 sites operator
            my_op = Ops[0].reshape(d,d,d,d,d,d,d,d)
            # Contraction
            AAAA = np.tensordot(AA, AA, axes=(3,0))
            ret = np.tensordot(my_op, AAAA, axes=([4,5,6,7],[1,2,3,4]))
            ret = np.tensordot(np.conj(AAAA), ret, axes=([1,2,3,4],[0,1,2,3]))
        elif Op_size == d**12: #6 sites operator
            my_op = Ops[0].reshape(d,d,d,d,d,d,d,d,d,d,d,d)
            # Contraction
            AAAA = np.tensordot(AA, AA, axes=(3,0))
            AAAAAA = np.tensordot(AA, AAAA, axes=(3,0))
            ret = np.tensordot(my_op, AAAAAA, axes=([6,7,8,9,10,11],[1,2,3,4,5,6]))
            ret = np.tensordot(np.conj(AAAAAA), ret, axes=([1,2,3,4,5,6],[0,1,2,3,4,5]))
        elif Op_size == d**16: #8 sites operator
            my_op = Ops[0].reshape(d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d)
            # Contraction
            AAAA = np.tensordot(AA, AA, axes=(3,0))
            AAAAAA = np.tensordot(AA, AAAA, axes=(3,0))
            AAAAAAAA = np.tensordot(AA, AAAAAA, axes=(3,0))
            ret = np.tensordot(my_op, AAAAAAAA, axes=([8,9,10,11,12,13,14,15],\
                                                    [1,2,3, 4, 5, 6, 7, 8]))
            ret = np.tensordot(np.conj(AAAAAAAA), ret, axes=([1,2,3,4,5,6,7,8],\
                                                             [0,1,2,3,4,5,6,7]))
    elif n_args == 2:
        Op1 = Ops[0]
        Op2 = Ops[1]
        # Contraction
        ret = np.tensordot(Op1, AA, axes=(1,1))
        ret = np.tensordot(Op2, ret, axes=(1,2))
        ret = np.tensordot(np.conj(AA), ret, axes=([1,2],[1,0]))
    else:
        print 'In transfer operator: wrong num of arguments'
        return 0

    # Transposition
    ret = np.transpose(ret, (0,2,1,3))

    return ret



######################################
#function: act_transfer_operator_left#
######################################

def act_transfer_operator_left(my_E, my_C):
    """Acts with the transfer operator my_E on the left on my_C

    Parameters
    ----------
    my_E :
    my_C :

    Returns
    -------
    my_Cp : = my_E(my_C) on the left

    """

    my_Cp = np.tensordot(my_E, my_C, axes=([0,1],[0,1]))

    return my_Cp

######################################
#   function: compute_corr_lengths   #
######################################

def compute_corr_lengths(my_E, N):
    """Computes the first N correlation lengths associated to the transfer
    operator my_E: -1/log(l_i), where l_i are the eigenvalues of my_E
    (left/right eigenvalues are the same so no worry about what
    action)

    Parameters
    ----------
    my_E : the (D,D,D,D) tensor
    
    Returns
    -------
    corr_length :

    TODO: Implement the passage of a linear operator instead of a tensor
          D**4 can be pretty big...

    """

    # Compute the second largest eigenvalue
    D = my_E.shape[0]
    E_mat = my_E.reshape((D*D,D*D))
    if D < 5: #if small do not use sparse and return all the eigs
        print ("D small, return all the eigenvalues")
        vals, v = linalg.eig(E_mat)
    else:
        vals = eigs(E_mat, k=N, which='LM', return_eigenvectors=False)

    if np.allclose(max(vals), 1.) == False:
        print ("In compute_corr_lengths: ERROR: largest eig not 1")
        return 0
    #print "In compute_corr_length: eigs: ", vals
    vals = np.sort(vals)
    # Compute the correlation lenghts
    corr_lengths = -1./np.log(vals)

    return corr_lengths

######################################
#function: act_transfer_operator_left#
######################################

def act_transfer_operator_left(my_E, my_C):
    """Acts with the transfer operator my_E on the left on my_C

    Parameters
    ----------
    my_E :
    my_C :

    Returns
    -------
    my_Cp : = my_E(my_C) on the left

    """

    my_Cp = np.tensordot(my_E, my_C, axes=([0,1],[0,1]))

    return my_Cp

################################################
#   function: compute_ent_entropy_mixed_repr   #
################################################

def compute_ent_entropy_mixed_repr(my_Lambda):
    """Computes the entanglement entropy of a state represented 
    by a mixed normalized representation: A...A.Lambda.B...B
    with A left normalized, B right normalized. Then the ent entropy
    for bipartition at last A is just given in terms of singular values
    of Lambda.

    Parameters
    ----------
    my_Lambda : 
    
    Returns
    -------
    ent_entropy :

    """
    
    if my_Lambda.ndim == 1: # means already sum of singular values
        s = my_Lambda
    elif my_Lambda.ndim == 2:
        U, s, Vh = linalg.svd(my_Lambda, full_matrices=False)
    else:
        print ("ERROR: in compute_ent_entropy_mixed_repr, my_Lambda has wrong ndim")
        return 0

    # Check that sum s_a^2 = 1 and real
    if np.isreal(s).all()!=True or np.isclose(sum(s**2),1.)==False:
        print ("ERROR: in compute_ent_entropy_mixed_repr, not good state...")
        return 0

    ent_entropy = -sum((s**2)*np.log(s**2))

    return ent_entropy

################################################
#          function: fidelity_wv               #
################################################

def fidelity_wv(my_psi1, my_psi2):
    """Computes 1 - overlap psi1,psi2

    Parameters
    ----------
    my_psi1 :
    my_psi2 :
     
    Returns
    -------
    f

    """
    
    # Normalize both vectors:
    psi1 = my_psi1/np.dot(my_psi1,my_psi1)
    psi2 = my_psi2/np.dot(my_psi2,my_psi2)
    if psi1.shape != psi2.shape:
        print ("In fidelity_wv: ERROR, psi1,psi2 different shapes")
        f = 0
    else:
        f = 1 - np.absolute(np.dot(psi1,psi2))

    return f

################################################
#        function: one_point_function          #
################################################

def one_point_function(A1, A2, Lambda_old, op, site):
    """Computes <psi|op(site)|psi>

    Parameters
    ----------
    op :
    site :
     
    Returns
    -------
    ret :

    """

    # even
    if np.mod(site,2) == 0: 
        AL = np.tensordot(A2, Lambda_old, axes=(2,0))
        tmp = np.tensordot(op, AL, axes=(1,1))
        ret = np.tensordot(np.conj(AL),tmp,axes=([0,1,2],[1,0,2]))
    # odd
    else:
        AL = np.tensordot(A2, Lambda_old, axes=(2,0))
        AAL = np.tensordot(A1, AL, axes=(2,0))
        tmp = np.tensordot(op, AAL, axes=(1,1))
        ret = np.tensordot(np.conj(AAL),tmp,axes=([0,1,2,3],[1,0,2,3]))

    return ret

################################################
#    function: bond_op_one_point_function      #
################################################

def bond_op_one_point_function(A1, A2, Lambda_old, B1, B2, op, site):
    """Computes <psi|op(site)|psi>. H.shape = (d^2, d^2)

    Parameters
    ----------
    ...
    op :
    site :
     
    Returns
    -------
    ret :

    """
    
    #reshape
    d = A1.shape[1]
    bond_op = op.reshape(d,d,d,d)
    # ev
    if np.mod(site,2) == 1: 
        if Lambda_old.ndim == 1:
            AAL = np.tensordot(A1,np.tensordot(A2,np.diag(Lambda_old),\
                                               axes=(2,0)), axes=(2,0))
        elif Lambda_old.ndim == 2:
            AAL = np.tensordot(A1,np.tensordot(A2,Lambda_old,axes=(2,0)), \
                               axes=(2,0))
        else:
            print ("ERROR: in bond_op_one_point, Lold has wrong ndim")

        tmp = np.tensordot(bond_op, AAL, axes=([2,3],[1,2]))
        ret = np.tensordot(np.conj(AAL), tmp, axes=([0,1,2,3],[2,0,1,3]))
     
    else: #odd
        if Lambda_old.ndim == 1:
            ALB = np.tensordot(A2, np.tensordot(np.diag(Lambda_old),B1,\
                                                axes=(1,0)), axes=(2,0))
        elif Lambda_old.ndim == 2:
            ALB = np.tensordot(A2, np.tensordot(Lambda_old,B1,axes=(1,0)), \
                               axes=(2,0))
        else:
            print ("ERROR: in bond_op_one_point, Lold has wrong ndim")
        tmp = np.tensordot(bond_op, ALB, axes=([2,3],[1,2]))
        ret = np.tensordot(np.conj(ALB), tmp, axes=([0,1,2,3],[2,0,1,3]))

    return ret

################################################
#    function: four_sites_op_one_point_function#
################################################

def four_sites_op_one_point_function(A1, A2, Lambda_old, B1, B2, op, site):
    """Computes <psi|op(site)|psi>. H.shape = (d^4, d^4)

    Parameters
    ----------
    ...
    op :
    site :
     
    Returns
    -------
    ret :

    """
    
    #reshape
    d = A1.shape[1]
    my_op = op.reshape(d,d,d,d,d,d,d,d)
    # odd
    if np.mod(site,2) == 1: 
        AA = np.tensordot(A1, A2, axes=(2,0))
        AAAA = np.tensordot(AA, AA, axes=(3,0))
        if Lambda_old.ndim == 1:
            AAAAL = np.tensordot(AAAA,np.diag(Lambda_old), axes=(5,0))
        elif Lambda_old.ndim == 2:
            AAAAL = np.tensordot(AAAA,Lambda_old, axes=(5,0))
        else:
            print ("ERROR: in four_sites_op_one_point, Lold has wrong ndim")

        tmp = np.tensordot(my_op, AAAAL, axes=([4,5,6,7],[1,2,3,4]))
        ret = np.tensordot(np.conj(AAAAL), tmp, axes=([0,1,2,3,4,5],\
                                                      [4,0,1,2,3,5]))
     
    else: #ev
        if Lambda_old.ndim == 1:
            ALB = np.tensordot(A2, np.tensordot(np.diag(Lambda_old),B1,\
                                                axes=(1,0)), axes=(2,0))
        elif Lambda_old.ndim == 2:
            ALB = np.tensordot(A2, np.tensordot(Lambda_old,B1,axes=(1,0)), \
                               axes=(2,0))
        else:
            print ("ERROR: in bond_op_one_point, Lold has wrong ndim")

        AA = np.tensordot(A2, A1, axes=(2,0))
        AAALB = np.tensordot(AA, ALB, axes=(3,0))

        tmp = np.tensordot(my_op, AAALB, axes=([4,5,6,7],[1,2,3,4]))
        ret = np.tensordot(np.conj(AAALB), tmp, axes=([0,1,2,3,4,5],\
                                                      [4,0,1,2,3,5]))

    return ret

################################################
#    function: six_sites_op_one_point_function#
################################################

def six_sites_op_one_point_function(A1, A2, Lambda_old, B1, B2, op, site):
    """Computes <psi|op(site)|psi>. H.shape = (d^6, d^6)

    Parameters
    ----------
    ...
    op :
    site :
     
    Returns
    -------
    ret :

    """
    
    #reshape
    d = A1.shape[1]
    my_op = op.reshape(d,d,d,d,d,d,d,d,d,d,d,d)
    # odd
    if np.mod(site,2) == 1: 
        AA = np.tensordot(A1, A2, axes=(2,0))
        AAAA = np.tensordot(AA, AA, axes=(3,0))
        AAAAAA = np.tensordot(AA, AAAA, axes=(3,0))
        if Lambda_old.ndim == 1:
            AAAAAAL = np.tensordot(AAAAAA,np.diag(Lambda_old), axes=(7,0))
        elif Lambda_old.ndim == 2:
            AAAAAAL = np.tensordot(AAAAAA,Lambda_old, axes=(7,0))
        else:
            print ("ERROR: in six_sites_op_one_point, Lold has wrong ndim")

        tmp = np.tensordot(my_op, AAAAAAL, axes=([6,7,8,9,10,11],[1,2,3,4,5,6]))
        ret = np.tensordot(np.conj(AAAAAAL), tmp, axes=([0,1,2,3,4,5,6,7],\
                                                        [6,0,1,2,3,4,5,7]))
     
    else: #ev
        if Lambda_old.ndim == 1:
            ALB = np.tensordot(A2, np.tensordot(np.diag(Lambda_old),B1,\
                                                axes=(1,0)), axes=(2,0))
        elif Lambda_old.ndim == 2:
            ALB = np.tensordot(A2, np.tensordot(Lambda_old,B1,axes=(1,0)), \
                               axes=(2,0))
        else:
            print ("ERROR: in six_sites_op_one_point, Lold has wrong ndim")

        AA = np.tensordot(A2, A1, axes=(2,0))
        AAALB = np.tensordot(AA, ALB, axes=(3,0))
        AAAAALB = np.tensordot(AA, AAALB, axes=(3,0))

        tmp = np.tensordot(my_op, AAAAALB, axes=([6,7,8,9,10,11],[1,2,3,4,5,6]))
        ret = np.tensordot(np.conj(AAAAALB), tmp, axes=([0,1,2,3,4,5,6,7],\
                                                        [6,0,1,2,3,4,5,7]))

    return ret

################################################
#          function: bond_en_and_S             #
################################################

def bond_en_and_S(h_bond, A, L, B, Lold, eta, X, Y):
    """Computes <psi|h_bond |psi>. for even/odd bonds and return their
    mean. Assume that left/right normalized unit cells.
    Further, computes S = entanglement entropy and returns

    Parameters
    ----------
    h_bond :
    A :
    L : 
    B :
    Lold : 
    eta :
    X :
    Y :
     
    Returns
    -------
    En_mean :
    S :

    """
    
    #reshape
    d = A.shape[1]
    bond_op = h_bond.reshape(d,d,d,d)
    # Define the rotate matrices of the unit cell
    A_t = np.tensordot(X, A, axes=(1,0))
    B_t = np.tensordot(B, Y, axes=(2,0))
    Lold_t = np.dot(X, np.dot(np.diag(Lold), Y))
    # Renormalize so that the max eigenvalue of transf operator = 1 
    L_t = L/np.sqrt(eta) 

    # Norm sqaured of wave function: trace of Lold_t.Lold_t^dagger
    norm2 = np.tensordot(Lold_t, np.conj(Lold_t), axes=([0,1],[0,1]))

    # even
    cell = np.tensordot(A_t, np.diag(L_t), axes=(2,0))
    cell = np.tensordot(cell, B_t, axes=(2,0))
    tmp = np.tensordot(bond_op, cell, axes=([2,3],[1,2]))
    En_ev = np.tensordot(np.conj(cell), tmp, axes=([0,1,2,3],[2,0,1,3]))

    # odd
    cell = np.tensordot(A_t, np.diag(L_t),axes=(2,0))
    cell = np.tensordot(cell, B, axes=(2,0))
    cell = np.tensordot(cell, np.diag(1/Lold), axes=(3,0))
    cell = np.tensordot(cell, A, axes=(3,0))
    cell = np.tensordot(cell, np.diag(L_t), axes=(4,0))
    cell = np.tensordot(cell, B_t, axes=(4,0))
    tmp = np.tensordot(bond_op, cell, axes=([2,3],[2,3]))
    En_odd = np.tensordot(np.conj(cell), tmp, axes=([0,1,2,3,4,5],\
                                                    [2,3,0,1,4,5]))

    En_mean = np.mean([En_ev,En_odd])/norm2

    # Compute entanglement entropy: the reduced density matrix is
    # Lold_t.Lold_t^dagger
    S = compute_ent_entropy_mixed_repr(Lold_t/np.sqrt(norm2))

    return En_mean, S

######################################
#    function: act_bond_operator     #
######################################

def act_bond_operator(U, psi):
    """Acts with the bond operator U on psi.
    Assumes psi has shape D,d,d,D and U (d,d,d,d)

    Parameters
    ----------
    U :
    psi :

    Returns
    -------
    ret : 

    """

    # Sanity check
    if psi.shape[0]!=psi.shape[3] or psi.shape[1]!=psi.shape[2]:
        print "In act_bond_operator: ERROR, wrongs psi.shape"
        return 0
        
    ret = np.tensordot(U, psi, axes=([2,3],[1,2]))
    ret = np.transpose(ret, (1,0,2,3))
    ret = np.transpose(ret, (2,1,0,3))

    return ret
