import numpy as np
import aux_functions
from scipy import integrate

######################################
#           Class: XXZ_H            #
######################################
class XXZ_H:
    """XXZ Hamiltonian: contains vars associated to the Hamiltonian:
    phys_dim, w (MPO), d_w (dim), two sites Hamiltonian.
    
    
    Parameters
    ----------
    gamma : assumed [0,pi/2]

    Returns
    -------
    object of this type.
    
    """

    # Local operators
    sz = np.array([[1,0],[0,-1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sp = np.array([[0,1],[0,0]]) #redundant
    sm = np.array([[0,0],[1,0]]) #redundant
    id_mat = np.eye(2)
    zero_mat = np.zeros([2,2])
    # E11= np.array([[1,0],[0,0]])
    # E12= np.array([[0,1],[0,0]])
    # E21= np.array([[0,0],[1,0]])
    # E22= np.array([[0,0],[0,1]])

    # Initialization
    def __init__(self, couplings):
        print("Inside class XXZ_H constructor")
        # Save model and couplings for later use
        self.model = 'XXZ'
        self.pars = couplings
        s = couplings['s']
        # Hamiltonian is always hermitian
        self.is_hermitian = True
        self.dtype = float
        Delta = np.cos(np.pi*s)
        self.phys_dim = 2
        # Hamiltonian for two sites, couplings used
        if s == 0:
            self.h_two_sites = np.kron(self.sp,self.sm) + \
                               np.kron(self.sm,self.sp) + \
                               Delta/2.*np.kron(self.sz,self.sz)
        else:
            self.h_two_sites = s/np.sin(np.pi*s)*(np.kron(self.sp,self.sm) + \
                                                  np.kron(self.sm,self.sp) + \
                                                  Delta/2.*np.kron(self.sz,self.sz))
        # MPO vars: left/right initial vectors vl,vr and W:
        # H = vl.W.W. ... .W.vr
        # Get rid of the prefactor in the definition of the MPO. Nothing changes.
        self.dim_W = 5 # aux MPO dimension
        # vl: 
        self.vl = np.array([self.zero_mat,\
                            self.sp,self.sm,Delta/2.*self.sz,self.id_mat])
        # Put in canonical form
        self.vl = np.transpose(self.vl, (1,0,2))
        # vr: 
        self.vr = np.array([self.id_mat,self.sm,self.sp,self.sz,self.zero_mat])
        # Already in canonical form
        # W:
        self.W = np.array([[self.id_mat,self.zero_mat,self.zero_mat,self.zero_mat,self.zero_mat], \
                           [self.sm,self.zero_mat,self.zero_mat,self.zero_mat,self.zero_mat], \
                           [self.sp,self.zero_mat,self.zero_mat,self.zero_mat,self.zero_mat], \
                           [self.sz,self.zero_mat,self.zero_mat,self.zero_mat,self.zero_mat], \
                           [self.zero_mat,\
                            self.sp,self.sm,Delta/2.*self.sz,self.id_mat]])
        # Put in canonical form, then action is on the last two
        # indices, like ordinary matrix. 
        self.W = np.transpose(self.W, (0, 2, 1, 3))
        # Here (D_W,d|D_W,d)

        # Example of action of W on v:
        #
        # v=np.zeros(6);v[0]=1;v=v.reshape(3,2)
        # np.tensordot(H.W,v,axes=([2,3],[0,1]))

        # Hamiltonian for a bond (same AB or BA). Same as two sites
        self.h_bond = self.h_two_sites
        # Energy
        self.energy_op = 0
        # S = T + \bar{T}
        #Einfty := <psi|h_i|psi> = ground state energy per site.
        if s==0:
            self.hinfty = 1/2.-2.*np.log(2) 
        else:
            gamma = np.pi*s
            f = lambda x,g : 1/np.cosh(np.pi*x)\
                /(np.cosh(2*g*x) - np.cos(g))
            E0Al = integrate.quad(f, 0, np.inf, args=(gamma,))[0]
            E0Al = 1/2.*np.cos(gamma) - 2*np.sin(gamma)*np.sin(gamma)*E0Al
            # E0Al is the energy of Eq. 2.49
            # http://snoelieputruiq.com/doc/Publ_26_Surface_exponents.pdf
            self.E0 = E0Al - Delta/2.
            self.hinfty = s/np.sin(np.pi*s)*self.E0 + Delta/2.*s/np.sin(np.pi*s)
        # Define h on 2 sites:
        self.S1 = self.h_two_sites-self.hinfty*np.kron(self.id_mat, self.id_mat)
        # Define h on 4 sites:
        hi_4 = aux_functions.mykron(self.h_two_sites,self.id_mat,self.id_mat)
        hip1_4 = aux_functions.mykron(self.id_mat,self.h_two_sites,self.id_mat)
        hip2_4 = aux_functions.mykron(self.id_mat,self.id_mat,self.h_two_sites)
        id_4 = aux_functions.mykron(self.id_mat,self.id_mat,self.id_mat,self.id_mat)
        self.S2 = hi_4+hip1_4-2*self.hinfty*id_4
        self.D2 = hi_4-hip1_4
        self.R2 = aux_functions.comm(hi_4, hip1_4)+\
                  aux_functions.comm(hip1_4, hip2_4)
        self.R1 = aux_functions.comm(hi_4, hip1_4)
        # Define h on 6 sites - even though it acts on 5, it is more convenient 6 due
        # to the A1 A2 structure of the MPS.
        hi_6 = aux_functions.mykron(self.h_two_sites,self.id_mat,self.id_mat, \
                                    self.id_mat, self.id_mat)
        hip1_6 = aux_functions.mykron(self.id_mat,self.h_two_sites,self.id_mat,\
                                      self.id_mat, self.id_mat)
        hip2_6 = aux_functions.mykron(self.id_mat,self.id_mat,self.h_two_sites,\
                                    self.id_mat, self.id_mat)
        hip3_6 = aux_functions.mykron(self.id_mat,self.id_mat,self.id_mat,\
                                      self.h_two_sites, self.id_mat)
        id_6 = aux_functions.mykron(self.id_mat,self.id_mat,self.id_mat,self.id_mat,\
                                    self.id_mat,self.id_mat)
        self.S3 = 1/2.*hi_6+hip1_6+1/2.*hip2_6-2*self.hinfty*id_6
        self.R3 = 1/2.*aux_functions.comm(hi_6, hip1_6)+\
                  aux_functions.comm(hip1_6, hip2_6)+\
                  1/2.*aux_functions.comm(hip2_6, hip3_6)
#        self.myid=id_4_sites
        #operator acting on 4 sites, i-1,i,i+1,i+2. (Actually 3,
        #but say 4 to be consistent with R which acts on 4 sites)
#        self.S = -np.pi*(him1+hi-2*self.hinfty*id_4_sites)
        # 2 sites

        # R = Txy = -i(T - \bar{T})
#        self.R = np.pi*(np.dot(him1,hi)-np.dot(hi,him1) \
#                 + np.dot(hi,hip1)-np.dot(hip1,hi))
        # 4 sites
#        self.comm = np.dot(him1,hi)-np.dot(hi,him1)


        # Invariant operators
        # TL on 2 sites
        ialpha = 1j*np.sin(np.pi*s)
        self.hinv = s/np.sin(np.pi*s)*(np.kron(self.sp,self.sm) + np.kron(self.sm,self.sp)+ \
                           1/2.*Delta*np.kron(self.sz,self.sz)+\
                           1/2.*ialpha*(np.kron(self.sz,self.id_mat)-\
                                   np.kron(self.id_mat,self.sz))-\
                           1/2.*Delta*np.kron(self.id_mat,self.id_mat))
        # and 4 sites
        hinvi_4 = np.kron(np.kron(self.hinv,self.id_mat),self.id_mat)
        hinvip1_4 = np.kron(self.id_mat,np.kron(self.hinv,self.id_mat))
        hinvip2_4 = np.kron(np.kron(self.id_mat,self.id_mat),self.hinv)
        id_4_sites=np.kron(self.id_mat,\
                           np.kron(np.kron(self.id_mat,self.id_mat),\
                                   self.id_mat))
        self.hinftyinv = s/np.sin(np.pi*s)*self.E0 
        self.S2inv = hinvi_4+hinvip1_4-2*self.hinftyinv*id_4
        self.D2inv = hinvi_4-hinvip1_4
