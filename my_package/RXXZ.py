import numpy as np
from scipy import integrate

######################################
#           Class: RXXZ_H            #
######################################
class RXXZ_H:
    """RXXZ Hamiltonian: contains vars associated to the Hamiltonian:
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
        print("Inside class RXXZ_H constructor")
        # Save model and couplings for later use
        self.model = 'RXXZ'
        self.pars = couplings
        gamma = couplings['gamma']
        # Hamiltonian is not hermitian for gamma \neq 0
        if gamma == 0:
            self.is_hermitian = True
            self.dtype = float
            ialpha = 0
        else:
            self.is_hermitian = False
            self.dtype = complex
            ialpha = 1j*np.sin(gamma)
        Delta = np.cos(gamma)
        self.phys_dim = 2
        # TL, everything is based on this
        self.TL = -1/2.*(2*np.kron(self.sp,self.sm) + 2*np.kron(self.sm,self.sp)+ \
                         Delta*np.kron(self.sz,self.sz)+\
                         ialpha*(np.kron(self.sz,self.id_mat)-\
                                 np.kron(self.id_mat,self.sz))-\
                         Delta*np.kron(self.id_mat,self.id_mat))
        # self.TL = 1/q*np.kron(self.E11,self.E22)+q*np.kron(self.E22,self.E11)\
        #           -(np.kron(self.E12,self.E21)+np.kron(self.E21,self.E12))

        # Hamiltonian for two sites, couplings used
        self.h_two_sites = -self.TL
        # MPO vars: left/right initial vectors vl,vr and W:
        # H = vl.W.W. ... .W.vr
        self.dim_W = 5 # aux MPO dimension
        # vl: 
        self.vl = np.array([ialpha/2.*self.sz-Delta/2.*self.id_mat,\
                            self.sp,self.sm,Delta/2.*self.sz,self.id_mat])
        # Put in canonical form
        self.vl = np.transpose(self.vl, (1,0,2))
        # vr: 
        self.vr = np.array([self.id_mat,self.sm,self.sp,self.sz,-ialpha/2.*self.sz])
        # Already in canonical form
        # W:
        self.W = np.array([[self.id_mat,self.zero_mat,self.zero_mat,self.zero_mat,self.zero_mat], \
                           [self.sm,self.zero_mat,self.zero_mat,self.zero_mat,self.zero_mat], \
                           [self.sp,self.zero_mat,self.zero_mat,self.zero_mat,self.zero_mat], \
                           [self.sz,self.zero_mat,self.zero_mat,self.zero_mat,self.zero_mat], \
                           [-Delta/2.*self.id_mat,\
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
        self.h_bond = -self.TL
        # Energy
        self.energy_op = 0
        # S = T + \bar{T}
        #Einfty := <psi|e_i|psi> = -ground state energy per site.
        if gamma==0:
            self.Einfty = 2.*np.log(2) 
        else:
            f = lambda x,g : 1/np.cosh(np.pi*x)\
                /(np.cosh(2*g*x) - np.cos(g))
            E0Al = integrate.quad(f, 0, np.inf, args=(gamma,))[0]
            E0Al = 1/2.*np.cos(gamma) - 2*np.sin(gamma)*np.sin(gamma)*E0Al
            # E0Al is the energy of Eq. 2.49
            # http://snoelieputruiq.com/doc/Publ_26_Surface_exponents.pdf
            self.E0 = E0Al - Delta/2.
            self.Einfty = -self.E0
        # Define TL on 4 sites:
        Eim1 = np.kron(np.kron(self.TL,self.id_mat),self.id_mat)
        Ei = np.kron(self.id_mat,np.kron(self.TL,self.id_mat))
        Eip1 = np.kron(np.kron(self.id_mat,self.id_mat),self.TL)
        id_4_sites=np.kron(self.id_mat,\
                           np.kron(np.kron(self.id_mat,self.id_mat),\
                                   self.id_mat))
        self.myid=id_4_sites
        #operator acting on 4 sites, i-1,i,i+1,i+2. (Actually 3,
        #but say 4 to be consistent with R which acts on 4 sites)
        if gamma==0:
            self.S = -(Eim1+Ei-2*self.Einfty*id_4_sites)
            # defined on 4 sites
            self.Snobd = self.S
            self.hn = -self.TL+self.Einfty*np.kron(self.id_mat, self.id_mat)
            self.hnobd = self.hn
            # R = Txy = -i(T - \bar{T})
            self.R = 1/(np.pi)*(np.dot(Eim1,Ei)-np.dot(Ei,Eim1) \
                                    + np.dot(Ei,Eip1)-np.dot(Eip1,Ei))
        else:
            #not implemented yet
            pref = -gamma/np.sin(gamma)
            self.S = pref*(Eim1+Ei-2*self.Einfty*id_4_sites)
            self.ss = 2*np.kron(self.sp,self.sm) + \
                      2*np.kron(self.sm,self.sp)+ \
                      Delta*np.kron(self.sz,self.sz)-\
                      Delta*np.kron(self.id_mat,self.id_mat)
            # defined on 4 sites
            self.Snobd = pref*(np.kron(np.kron(self.ss, self.id_mat), self.id_mat)+\
                         np.kron(np.kron(self.id_mat, self.ss), self.id_mat) -\
                               2*self.Einfty*id_4_sites)
            self.hn = -self.TL-self.Einfty*np.kron(self.id_mat, self.id_mat)
            self.hnobd = 1/2.*(2*np.kron(self.sp,self.sm) + 2*np.kron(self.sm,self.sp)+ \
                               Delta*np.kron(self.sz,self.sz)+\
                               Delta*np.kron(self.id_mat,self.id_mat))-\
                               self.Einfty*np.kron(self.id_mat, self.id_mat)
            # R = Txy = -i(T - \bar{T})
            pref = 1/np.pi*(gamma/np.sin(gamma))**2
            self.R = pref*(np.dot(Eim1,Ei)-np.dot(Ei,Eim1) \
                           + np.dot(Ei,Eip1)-np.dot(Eip1,Ei))

        # Elementary row to row transfer matrix
        self.r2r_T = np.kron(self.id_mat,self.id_mat)+self.TL
