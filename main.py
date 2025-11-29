from common_modules import np, os, plt
from src.local_coordinates import *
from src.shape_function import *
from src.mesh_operations import *
from src.integration import *

if __name__=="__main__":
    testcase = os.path.join("msh", "ductQuadMesh.msh")

    conf = {} # intialization of config dict

    ## FREQUENCY INDEPENDENT CONFIG DATA ##
    # props of air
    conf['c'] = 343. # [m/s] speed of sound
    conf['rho'] = 1.2 # [kg/m^3] density

    # boundary conditions
    conf['IntExt'] = 1 # interior (1) or exterior (-1) problem
    conf['vBC_x0'] = 0.001 # [m/s] structural velocity
    conf['YBC_xMax'] = 1./(conf['rho']*conf['c']) # [m^3/Pas] acoustic admittance (to do: frequency dependent)

    # integration properties
    conf['DiscElementOrder'] = 1 # order of discontinuous elements
    conf['DiscIntOrder'] = 1 # order of interpolation functions
    conf['DiscAlpha'] = 0.42 # optimal position of iso-parametric coord in zeros of lin. lagrangian polynomials, only for pure Neumann prob to do: script for varible freq dependent alpha

    ## FREQUENCY DEPENDENT CONFIG DATA##
    conf['frq'] = 100 # [Hz] frequency
    conf['omega'] = 2*np.pi*conf['frq'] # [1/s] circular frequency

    # props of air
    conf['lambda'] = conf['c']/conf['frq'] # [m] wave length
    conf['k'] = conf['omega']/conf['c'] # [1/m] wave number

    ## READ INPUT ##
    # import and check geometric mesh, calc normals and tangentials ###
    geoA = readMesh(testcase,conf['IntExt']) # returns dic

    # map to parameter space, convert to discontinuous mesh ###
    eta = evalLocCoord(conf['DiscAlpha'],conf['DiscElementOrder']) # isoparametric space coord for 4-node quadrangle
    geoB = discMesh(geoA,eta) # returns dic

    ## ===== PROCESSING ===== ##
    ## INITIALIZE ##
    G = np.zeros((geoB['numNodes'],geoB['numNodes']),dtype=complex) # single layer potential
    H = np.zeros((geoB['numNodes'],geoB['numNodes']),dtype=complex) # double layer potential
    c0 = np.zeros((1,geoB['numNodes']),dtype=float) # integral free term for every cp
    sk = 1j*conf['rho']*conf['c']*conf['k'] # multiplication factor
    # initialize counters
    countSing = 0
    countReg = 0 

    ## MATRIX ASSEMBLY ##
    for cp in range(geoB['numNodes']): # collocation point loop
        for el in range(geoB['numElements']): # element loop
            
            #  singularity check
            DOF = np.where(geoB['elements'][el,:]==cp)[0] # DOF goes from 0 to 3 
            singular = len(DOF) > 0                       # check if DOF array is empty, if not --> singularIntegration 

            #  choose then the relevant integration technique
            if singular:
                #  singular integration via polar coordinate transformation
                g_el,h_el,c_el = singularIntegration(geoA,geoB,conf,cp,el,DOF[0],eta)
                countSing += 1
            else: 
                # Regular Gauss integration
                g_el,h_el,c_el = regularIntegration(geoA,geoB,conf,cp,el)
                countReg += 1

            # fill matrices with element vectors
            G[cp,geoB['numNodPerEl']*el:geoB['numNodPerEl']*(el+1)] = sk*g_el.T
            H[cp,geoB['numNodPerEl']*el:geoB['numNodPerEl']*(el+1)] = h_el.T
            c0[0,cp] += c_el # sum up integral free term for every cp over all elements (=1/0/0.5 inside domain/outside domain/on boundary)
        
        outputMsgC0 = 'Sum over ' + str(geoB['numElements']) + ' elements at collocation point (' + str(cp+1) + '/' + str(geoB['numNodes']) + '): c0 =' + str(np.round(c0[0,cp],8))
        print(outputMsgC0) # output message

    H += np.diag(c0[0]) # add sum of integral free term to diagonal entries (dirac function =1, otherwise 0)

    
    ## BOUNDARY CONDITIONS ##
    # initialize Robin Boundary boundary condition
    vs = np.zeros((geoB['numNodes'],1),dtype=float) 
    Y = np.zeros((1,geoB['numNodes']),dtype=float)
    # get boundary indices and unique x coord+according index (for BC/plotting/validation)
    fBid = np.where(np.absolute(geoB['nodes'][:,0]- np.min(geoB['nodes'][:,0]))<10**(-8))[0] # Node indices front of duct
    eBid = np.where(np.absolute(geoB['nodes'][:,0]- np.max(geoB['nodes'][:,0]))<10**(-8))[0] # Node indices end side of duct
    # fill in vector entries
    vs[fBid] = conf['vBC_x0']
    Y[0,eBid] = conf['YBC_xMax'] # column vector for velocity
    Y = np.diag(Y[0]) # diagonal matriz for admittance

    ## SOLVER ##
    # BEM equation: (H-GY)p=Gvs
    b = np.matmul(G,vs) # multiplication otherwise elementwise, =- works
    A = H - np.matmul(G,Y)
    p = np.linalg.solve(A, b)

    ## ===== POSTPROCESSING ===== ##
    rBid = np.where(np.absolute(geoB['nodes'][:,1]- np.min(geoB['nodes'][:,1]))<10**(-8))[0] # Node indices right side of duct
    evalx,tmpIdx = np.unique(np.round(geoB['nodes'][rBid,0],6),return_index=True)
    idx = rBid[tmpIdx]

    p_abs = np.absolute(p)

    # 2D line plot for all frequencies 
    figName = 'Acoustic pressure'
    plt.plot(evalx,p[idx,:].real,'r',evalx,p[idx,:].imag,'b',evalx,p_abs[idx,:],'g')
    plt.xlabel('duct length x [m]')
    plt.ylabel('acoustic pressure pa [Pa]')
    plt.title(figName)
    plt.show()

