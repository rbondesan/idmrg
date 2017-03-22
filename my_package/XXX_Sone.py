import numpy as np
from scipy import integrate

######################################
#           Class: XXX_Sone          #
######################################
class XXX_Sone_H:
    """XXX spin 1 Hamiltonian: contains vars associated to the
    Hamiltonian: phys_dim, w (MPO), d_w (dim), two sites Hamiltonian.
    
    
    Parameters
    ----------
    None, it is the critical Hamiltonian with limit SU2 level 2

    Returns
    -------
    object of this type.

    """

    # Initialization
    def __init__(self, couplings):

        print("Inside class XXX_Sone constructor")

        # Local operators
        dim = 3
        sp = np.sqrt(2)*np.array([[0,1,0],[0,0,1],[0,0,0]])
        sm = np.sqrt(2)*np.array([[0,0,0],[1,0,0],[0,1,0]])
        s3 = np.array([[1,0,0],[0,0,0],[0,0,-1]])
        spsm = np.dot(sp,sm)
        smsp = np.dot(sm,sp)
        s3s3 = np.dot(s3,s3)
        spsp = np.dot(sp,sp)
        smsm = np.dot(sm,sm)
        sps3 = np.dot(sp,s3)
        s3sp = np.dot(s3,sp)
        sms3 = np.dot(sm,s3)
        s3sm = np.dot(s3,sm)
        id_mat = np.eye(dim)
        zero_mat = np.zeros([dim,dim])

        # Save model and couplings for later use.  pars is empty, just
        # for book keeping with names of files to save data
        self.model = 'XXX_Sone'
        self.pars = couplings
        # Hamiltonian is hermitian 
        self.is_hermitian = True
        self.dtype = float
        self.phys_dim = dim
        # Quadratic Casimir on two sites
        self.Cas = 1/2.*(np.kron(sp,sm) + np.kron(sm,sp)) + np.kron(s3,s3)
        # Hamiltonian for two sites
        self.h_two_sites = 1/2./np.pi*(self.Cas - np.dot(self.Cas,self.Cas) + \
                                       3*np.eye(self.phys_dim**2))
        self.h_two_sites2 = 1/2./np.pi*(1/2.*(np.kron(sm,sp)+np.kron(sp,sm)) +\
                                        np.kron(s3,s3)-(1/4.*np.kron(spsm,smsp) + \
                                                        1/4.*np.kron(smsp,spsm) + \
                                                        1/2.*np.kron(sps3,sms3) + \
                                                        1/2.*np.kron(sms3,sps3) + \
                                                        1/2.*np.kron(s3sp,s3sm) + \
                                                        1/2.*np.kron(s3sm,s3sp) + \
                                                        np.kron(s3s3,s3s3)) + \
                                        3*np.kron(id_mat,id_mat))

        # MPO vars: left/right initial vectors vl,vr and W:
        # H = vl.W.W. ... .W.vr
        self.dim_W = 14 # aux MPO dimension
        # vl: 
        self.vl = 1/2./np.pi*np.array([3.*id_mat, sm, sp, s3, smsm, smsp, sms3, spsp, \
                                       spsm, sps3, s3sm, s3sp, s3s3, id_mat])
        # Put in canonical form
        self.vl = np.transpose(self.vl, (1,0,2))
        # vr: 
        self.vr = np.array([id_mat, 1/2.*sp, 1/2.*sm, s3, -1/4.*spsp, -1/4.*spsm, -1/2.*sps3, \
                            -1/4.*smsm, -1/4.*smsp, -1/2.*sms3, -1/2.*s3sp, -1/2.*s3sm, -s3s3, \
                            zero_mat])
        # Already in canonical form
        # W:
        self.W = np.array([ \
                           # row 1
                           [id_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 2
                           [1/2.*sp, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 3
                           [1/2.*sm, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 4
                           [s3, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 5
                           [-1/4.*spsp, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 6
                           [-1/4.*spsm, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 7
                           [-1/2.*sps3, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 8
                           [-1/4.*smsm, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 9
                           [-1/4.*smsp, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 10
                           [-1/2.*sms3, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 11
                           [-1/2.*s3sp, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 12
                           [-1/2.*s3sm, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 13
                           [-s3s3, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, \
                            zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat], \
                           # row 14
                           [3*id_mat, sm, sp, s3, smsm, smsp, sms3, \
                            spsp, spsm, sps3, s3sm, s3sp, s3s3, id_mat]])
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
        self.Einfty = -0.159110040863 #2.*np.log(2) 
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
        id_4_sites = np.kron(id_mat,np.kron(id_mat,np.kron(id_mat,id_mat)))
        hnm1 = np.kron(np.kron(self.h_two_sites,id_mat),id_mat)
        hn   = np.kron(np.kron(id_mat, self.h_two_sites),id_mat)
        hnp1 = np.kron(np.kron(id_mat,id_mat),self.h_two_sites)
        self.S = -(hnm1 + hn - 2*self.Einfty*id_4_sites)
        # R = T - \bar{T}
        self.R = np.dot(hnm1,hn)-np.dot(hn,hnm1) \
                 + np.dot(hn,hnm1)-np.dot(hnm1,hn)
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
