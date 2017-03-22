import numpy as np

######################################
#           Class: IsingH            #
######################################
class Ising_H:
    """Ising Hamiltonian: contains vars associated to the Hamiltonian:
    phys_dim, w (MPO), d_w (dim), two sites Hamiltonian.

    
    Parameters
    ----------
    couplings : Couplings of the Hamiltonian

    Returns
    -------
    object of this type.
    
    """

    # Local operators
    sz = np.array([[1,0],[0,-1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    isy = np.array([[0,1],[-1,0]])
    id_mat = np.eye(2)
    zero_mat = np.zeros([2,2])

    # Initialization
    def __init__(self, couplings):
        print("Inside class Ising_H constructor")
        # Save model and couplings for later use
        self.model = 'Ising'
        self.pars = couplings
        # Hamiltonian is hermitian 
        self.is_hermitian = True
        self.dtype = float
        self.phys_dim = 2
        # Hamiltonian for two sites, couplings used
        self.h_two_sites = -couplings['J']*np.kron(self.sz,self.sz) \
                           -couplings['h']*(np.kron(self.sx,self.id_mat) \
                                            + np.kron(self.id_mat,self.sx))
        # MPO vars: left/right initial vectors vl,vr and W:
        # H = vl.W.W. ... .W.vr
        self.dim_W = 3 # aux MPO dimension
        ratio_coupl = couplings['h']/couplings['J']
        # vl: 
        self.vl = np.array([-couplings['h']*self.sx, -couplings['J']*self.sz,\
                            -couplings['J']*self.id_mat])
        # Put in canonical form
        self.vl = np.transpose(self.vl, (1,0,2))
        # vr:
        self.vr = np.array([self.id_mat, self.sz, ratio_coupl*self.sx])
        # Already in canonical form
        # W:
        self.W = np.array([[self.id_mat,self.zero_mat,self.zero_mat], \
                           [self.sz, self.zero_mat, self.zero_mat], \
                           [ratio_coupl*self.sx, self.sz, self.id_mat]])
        # Put in canonical form, then action is on the last two
        # indices, like ordinary matrix. 
        self.W = np.transpose(self.W, (0, 2, 1, 3))
        # Here (D_W,d|D_W,d)

        # Example of action of W on v:
        #
        # v=np.zeros(6);v[0]=1;v=v.reshape(3,2)
        # np.tensordot(H.W,v,axes=([2,3],[0,1]))

        # Hamiltonian for a bond (same AB or BA). Note different from
        # h_2_sites by h/2 vs h
        self.myid = np.kron(self.id_mat,self.id_mat)
        self.h_bond = -couplings['J']*np.kron(self.sz,self.sz) \
                      -couplings['h']/2.*(np.kron(self.sx,self.id_mat) \
                                          + np.kron(self.id_mat,self.sx))
        # Energy
        self.energy_op = 2*np.kron(self.sz,self.sz) - np.kron(self.sx,self.id_mat) \
                         - np.kron(self.id_mat,self.sx)

        # S = T + \bar{T}
        E0 = -1.27323954474 #ground state energy per site
        self.Einfty = -E0 #ground state energy per site
        self.Sav = np.pi*(self.h_bond - E0*np.kron(self.id_mat,self.id_mat))
        self.S = -1/4*np.pi*(np.kron(self.sz,self.sz) \
                             + np.kron(self.sx,self.id_mat) \
                             + E0*np.kron(self.id_mat,self.id_mat))
        self.Salt = -1/4*np.pi*(np.kron(self.sz,self.sz) \
                             + np.kron(self.sx,self.id_mat) \
                             + E0*np.kron(self.id_mat,self.id_mat))

        # R = T - \bar{T}
#        self.R = np.pi/2.*(np.kron(self.sz,self.sy)-np.kron(self.sy,self.sz))
        self.R = np.pi/16.*(np.kron(self.sz,self.isy)-np.kron(self.isy,self.sz))
