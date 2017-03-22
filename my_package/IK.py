import numpy as np
from scipy import integrate

######################################
#           Class: IK                #
######################################
class IK_H:
    """Izergin Korepin Hamiltonian: contains vars associated to the
    Hamiltonian: phys_dim, w (MPO), d_w (dim), two sites Hamiltonian.
    
    
    Parameters
    ----------
    eta :

    Returns
    -------
    object of this type.

    """

    # Initialization
    def __init__(self, couplings):

        print("Inside class IK constructor")

        # Local operators
        # Unit matrices
        dim = 3
        def E(i,j):
            ret = np.zeros([dim,dim])
            # convert to 0,...,2 indices
            ret[i-1,j-1] = 1    
            return ret
        id_mat = np.eye(dim)
        zero_mat = np.zeros([dim,dim])

        # Save model and couplings for later use. 
        self.model = 'XXX_Sone'
        self.pars = couplings
        eta = couplings['eta']
        # Hamiltonian is not hermitian for gamma \neq 0
        if eta == 0:
            self.is_hermitian = True
            self.dtype = float
        else:
            self.is_hermitian = False
            self.dtype = complex
        w = np.exp(2*eta)-np.exp(-2*eta)
        self.phys_dim = dim
        # Two sites:
        self.B = np.array([[np.exp(2*eta), 0, 0, 0, 0, 0, 0, 0, 0], \
                           [0, 0, 0, 1, 0, 0, 0, 0, 0], \
                           [0, 0, 0, 0, 0, 0, np.exp(-2*eta), 0, 0], \
                           [0, 1, 0, w, 0, 0, 0, 0, 0], \
                           [0, 0, 0, 0, 1, 0, np.exp(-eta)*w, 0, 0], \
                           [0, 0, 0, 0, 0, 0, 0, 1, 0], \
                           [0, 0, np.exp(-2*eta), 0, np.exp(-eta)*w, 0, (1 - np.exp(-2*eta))*w, \
                            0, 0], \
                           [0, 0, 0, 0, 0, 1, 0, w, 0], \
                           [0, 0, 0, 0, 0, 0, 0, 0, np.exp(2*eta)]])
        self.Binv = np.array([[np.exp(-2*eta), 0, 0, 0, 0, 0, 0, 0, 0], \
                              [0, -w, 0, 1, 0, 0, 0, 0, 0], \
                              [0, 0, -1 + np.exp(4*eta) - w, 0, -np.exp(eta)*w, 0, np.exp(2*eta), \
                               0, 0], \
                              [0, 1, 0, 0, 0, 0, 0, 0, 0], \
                              [0, 0, -np.exp(eta)*w, 0, 1, 0, 0, 0, 0], \
                              [0, 0, 0, 0, 0, -w, 0, 1, 0], \
                              [0, 0, np.exp(2*eta), 0, 0, 0, 0, 0, 0], \
                              [0, 0, 0, 0, 0, 1, 0, 0, 0], \
                              [0, 0, 0, 0, 0, 0, 0, 0, np.exp(-2*eta)]])
        self.h_two_sites = np.exp(3*eta)*self.B + np.exp(-3*eta)*self.Binv

        # MPO vars: left/right initial vectors vl,vr and W:
        # H = vl.W.W. ... .W.vr
        self.dim_W = 21 # aux MPO dimension
        # Matrix elements
        w1111 = np.exp(5*eta) + np.exp(-5*eta)
        w3333 = w1111
        w1122 = -np.exp(-3*eta)*w;
        w2233 = w1122;
        w2112 = np.exp(-3*eta) + np.exp(3*eta);
        w1221 = w2112;
        w3223 = w2112;
        w2222 = w2112;
        w2332 = w2112;
        w1232 = -np.exp(-2*eta)*w;
        w2123 = w1232;
        w1331 = np.exp(-eta) + np.exp(eta);
        w3113 = w1331;
        w2211 = np.exp(3*eta)*w;
        w3322 = w2211;
        w2321 = np.exp(2*eta)*w;
        w3212 = w2321;
        w1133 = -np.exp(-3*eta) + np.exp(eta) - np.exp(-3*eta)*w;
        w3311 = -np.exp(eta)*w + np.exp(3*eta)*w;
        # vl: 
        self.v1 = np.array([E(1, 1), E(1, 1), E(1, 1), E(1, 2), E(1, 2), E(1, 3), \
                            E(2, 1), E(2, 1), E(2, 2), E(2, 2), E(2, 2), E(2, 3), \
                            E(2, 3), E(3, 1), E(3, 2), E(3, 2), E(3, 3), E(3, 3), E(3, 3)])
        self.vl = np.append(np.array([zero_mat]), self.v1, axis=0)
        self.vl = np.append(self.vl, np.array([id_mat]), axis=0)
        # Put in canonical form
        self.vl = np.transpose(self.vl, (1,0,2))
        # vr: 
        self.v2 = np.array([w1111*E(1, 1), w1122*E(2, 2), w1133*E(3, 3), w1221*E(2, 1), \
                            w1232*E(3, 2), w1331*E(3, 1), w2112*E(1, 2), w2123*E(2, 3), \
                            w2211*E(1, 1), w2222*E(2, 2), w2233*E(3, 3), w2321*E(2, 1), \
                            w2332*E(3, 2), w3113*E(1, 3), w3212*E(1, 2), w3223*E(2, 3), \
                            w3311*E(1, 1), w3322*E(2, 2), w3333*E(3, 3)])
        self.vr = np.append(np.array([id_mat]), self.v2, axis=0)
        self.vr = np.append(self.vr, np.array([zero_mat]), axis=0)
        # Already in canonical form
        # W: construct by blocks
        self.W = np.kron(np.ones((self.dim_W-1,self.dim_W-1)), zero_mat).reshape(self.dim_W-1,\
                                                                            self.dim_W-1,dim,dim)
        self.W = np.hstack((np.append(np.array([id_mat]), self.v2, axis=0).reshape(self.dim_W-1,\
                                                                           1,dim,dim), self.W))
        self.W = np.vstack((self.W, self.vl.transpose(1,0,2).reshape(1,self.dim_W,dim,dim)))
        # Put in canonical form, then action is on the last two
        # indices, like ordinary matrix. 
        self.W = np.transpose(self.W, (0, 2, 1, 3))
        # Here (D_W,d|D_W,d)

        # Hamiltonian for a bond (same AB or BA). Same as two sites
        self.h_bond = self.h_two_sites

        # Operators to be written!!

        # # Energy
        # self.energy_op = 0
        # # S = T + \bar{T}
        # #Einfty := <psi|e_i|psi> = -ground state energy per site.
        # if gamma==0:

        # see http://iopscience.iop.org/article/10.1088/0305-4470/25/11/016/pdf
        # eq. 6.6
        self.Einfty = 0 #2.*np.log(2) 

        # else:
        #     f = lambda x,g : 1/np.cosh(np.pi*x)\
        #         /(np.cosh(2*g*x) - np.cos(g))
        #     E0Al = integrate.quad(f, 0, np.inf, args=(gamma,))[0]
        #     E0Al = 1/2.*np.cos(gamma) - 2*np.sin(gamma)*np.sin(gamma)*E0Al
        #     # E0Al is the energy of Eq. 2.49
        #     # http://snoelieputruiq.com/doc/Publ_26_Surface_exponents.pdf
        #     self.E0 = E0Al - Delta/2.
        #     self.Einfty = -self.E0
        # # Define TL on 4 sites:
        # Eim1 = np.kron(np.kron(self.TL,self.id_mat),self.id_mat)
        # Ei = np.kron(self.id_mat,np.kron(self.TL,self.id_mat))
        # Eip1 = np.kron(np.kron(self.id_mat,self.id_mat),self.TL)
        # id_4_sites=np.kron(self.id_mat,\
        #                    np.kron(np.kron(self.id_mat,self.id_mat),\
        #                            self.id_mat))
        # self.myid=id_4_sites
        # #operator acting on 4 sites, i-1,i,i+1,i+2. (Actually 3,
        # #but say 4 to be consistent with R which acts on 4 sites)
        # if gamma==0:
        #     self.S = -(Eim1+Ei-2*self.Einfty*id_4_sites)
        #     self.St = Eim1
        #     # R = T - \bar{T}
        #     self.R = 1/(np.pi*1j)*(np.dot(Eim1,Ei)-np.dot(Ei,Eim1) \
        #                            + np.dot(Ei,Eip1)-np.dot(Eip1,Ei))
        # else:
        #     #not implemented yet
        #     pref = -gamma/np.sin(gamma)
        #     self.S = pref*(Eim1+Ei-2*self.Einfty*id_4_sites)
        #     self.St = Eim1
        #     # R = T - \bar{T}
        #     pref = np.pi*(gamma/(np.pi*np.sin(gamma)))**2; pref=pref/1j
        #     self.R = pref*(np.dot(Eim1,Ei)-np.dot(Ei,Eim1) \
        #                    + np.dot(Ei,Eip1)-np.dot(Eip1,Ei))

        # # Elementary row to row transfer matrix
        # self.r2r_T = np.kron(self.id_mat,self.id_mat)+self.TL
