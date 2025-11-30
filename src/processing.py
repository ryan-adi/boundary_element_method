from common_modules import np
from .integration import *

class Processing:
    def __init__(self, conf, geoA, eta, geoB):
        self.conf = conf
        self.geoA = geoA
        self.eta = eta
        self.geoB = geoB

        self.G = np.zeros((self.geoB['numNodes'],self.geoB['numNodes']),dtype=complex) # single layer potential
        self.H = np.zeros((self.geoB['numNodes'],self.geoB['numNodes']),dtype=complex) # double layer potential
        self.c0 = np.zeros((1,self.geoB['numNodes']),dtype=float) # integral free term for every cp
        self.sk = 1j*self.conf['density']*self.conf['speedOfSound']*self.conf['waveNumber'] # multiplication factor
        self.vs = np.zeros((self.geoB['numNodes'],1),dtype=float) 
        self.Y = np.zeros((1,self.geoB['numNodes']),dtype=float)

    def assemble_matrix(self):
        # initialize counters
        countSing = 0
        countReg = 0 

        for cp in range(self.geoB['numNodes']): # collocation point loop
            for el in range(self.geoB['numElements']): # element loop
                
                #  singularity check
                DOF = np.where(self.geoB['elements'][el,:]==cp)[0] # DOF goes from 0 to 3 
                singular = len(DOF) > 0                       # check if DOF array is empty, if not --> singularIntegration 

                #  choose then the relevant integration technique
                if singular:
                    #  singular integration via polar coordinate transformation
                    g_el,h_el,c_el = singularIntegration(self.geoA, self.geoB, self.conf,cp,el,DOF[0], self.eta)
                    countSing += 1
                else: 
                    # Regular Gauss integration
                    g_el,h_el,c_el = regularIntegration(self.geoA, self.geoB, self.conf,cp,el)
                    countReg += 1

                # fill matrices with element vectors
                self.G[cp,self.geoB['numNodPerEl']*el:self.geoB['numNodPerEl']*(el+1)] = self.sk * g_el.T
                self.H[cp,self.geoB['numNodPerEl']*el:self.geoB['numNodPerEl']*(el+1)] = h_el.T
                self.c0[0,cp] += c_el # sum up integral free term for every cp over all elements (=1/0/0.5 inside domain/outside domain/on boundary)
            
            outputMsgC0 = 'Sum over ' + str(self.geoB['numElements']) + ' elements at collocation point (' + str(cp+1) + '/' + str(self.geoB['numNodes']) + '): c0 =' + str(np.round(self.c0[0,cp],8))
            print(outputMsgC0) # output message

        self.H += np.diag(self.c0[0]) # add sum of integral free term to diagonal entries (dirac function =1, otherwise 0)

    def boundary_conditions(self):
       
        # get boundary indices and unique x coord+according index (for BC/plotting/validation)
        fBid = np.where(np.absolute(self.geoB['nodes'][:,0]- np.min(self.geoB['nodes'][:,0]))<10**(-8))[0] # Node indices front of duct
        eBid = np.where(np.absolute(self.geoB['nodes'][:,0]- np.max(self.geoB['nodes'][:,0]))<10**(-8))[0] # Node indices end side of duct
        
        # fill in vector entries
        self.vs[fBid] = self.conf['vBC_x0']
        self.Y[0,eBid] = self.conf['YBC_xMax'] # column vector for velocity
        self.Y = np.diag(self.Y[0]) # diagonal matrix for admittance

    def solve(self):
        # BEM equation: (H-GY)p=Gvs
        b = np.matmul(self.G, self.vs) 
        A = self.H - np.matmul(self.G, self.Y)
        p = np.linalg.solve(A, b)

        return p
