from common_modules import np
from .shape_function import NiX, NigradX, NiXDiscSurf


def EvalDecisionCriterion(nodesPerElem,coordElemNodes,coordCollocPoint): 
    """
    Evaluate the center-point of the element
    Determine the distance R between the collocation point and the center of the element
    Determine the element length h via the distance of two adjacent element nodes
    Remark: This works only if the nodes are numbered clockwise or anti-clockwise
    """
    coordCenterElem = np.sum(coordElemNodes,axis=0)/nodesPerElem
    R = np.linalg.norm(coordCenterElem - coordCollocPoint)
    h1 = np.linalg.norm(coordElemNodes[1,:] - coordElemNodes[0,:])
    h2 = np.linalg.norm(coordElemNodes[2,:] - coordElemNodes[1,:])
    h = np.maximum(h1,h2)
    D = 2*R/h
    
    return D

def NumberOfGaussPoints(D): 
    """
    Decide on number of Gauss points for integration on (non-singular)
    boundary elements based on the relative distance D to the collocation
    point
    """
    assert D >.05, 'Warning: Relative distance is too small. Please use a finer mesh.'
    
    if D > 20.:
        ngp = 4
    elif 12.<D and D<=20.:
        ngp = 6
    elif 7.<D and D<=12.:
        ngp = 8
    elif 3.6<D and D<=7:
        ngp = 10
    elif 1.2<D and D<=3.6:
        ngp = 12
    elif .4<D and D<=1.2:
        ngp = 24
    elif .05<D and D<=.4:
        ngp = 36
    return ngp

def polarTrafoTri(etaCP,etaGP,numTri): 
    """
    Transformation to polar coordinates to resolve weak singularity
    Input:  etaCP == (2,1) local coordinates of Collocation points
            etaGP == (2,1) local coordinates of Gauss points
            numTri == (int) triangle number, i.e a number from 0 to 3
    Output: etaTri == (2,1) local coordinates 
            JdetTri== (float) totaljacobian  
    """
    etaCPx,etaCPy = [*etaCP]
    etaGPx,etaGPy = [*etaGP]
    
    # height of triangles (distances of the singular point to the edges of quad.)
    H = np.abs([etaCPy+1,1-etaCPx,1-etaCPy,etaCPx+1])
    # angles in the corner of the triangle 
    # based on angles defined in S4 slide 8 
    thetaList = np.zeros((5,),dtype=float)              # initialize
    thetaList[0] = -np.pi + np.arctan(H[0]/H[3])
    thetaList[1] = - np.arctan(H[0]/H[1])
    thetaList[2] = np.arctan(H[2]/H[1])
    thetaList[3] = np.pi - np.arctan(H[2]/H[3])
    thetaList[4] = np.pi + np.arctan(H[0]/H[3])

    #  linear interpolation of theta
    theta_min = thetaList[numTri] 
    theta_max = thetaList[numTri+1] 
    theta = theta_min + (theta_max - theta_min) * (etaGPx+1) / 2        # see S4 slide 8

    #  linear interpolation of rho
    rho_min = 0                                           
    rho_max = H[numTri] / np.sin(np.pi*(numTri/2)-theta)  # rho_max as function of H[numTri] and theta
    rho = rho_min + (rho_max - rho_min) * (etaGPy+1) / 2  # see S4 slide 9
    
    #  evaluation of triangle coordinates
    etaTri = np.array(etaCP+[rho*np.cos(theta),rho*np.sin(theta)])      

    #  determinant for triangle transformation
    detQuadTri= rho                                             # see S4 slide 7

    # determinante for backtransformation to quad. system
    detTriQuad = abs(rho_max * (theta_max-theta_min)) / 4       # see S4 slide 10
    
    #  total determinant    
    JdetTri = detQuadTri * detTriQuad                           # multiply both determinants

    return etaTri,JdetTri 

def regularIntegration(A,B,conf,cp,el): 
    """
    regular integration routine to evaluate system matrices 
    of isentropic collocation BEM
    """
    elindices = A['elements'][el,:] # cont node ids of the current element
    elCoord = A['nodes'][elindices, :] # cont nodal coordinates of the current element
    cpCoord = B['nodes'][cp,:] # cp coordinates
    relDist = EvalDecisionCriterion(A['numNodPerEl'],elCoord,cpCoord)
    numGP = NumberOfGaussPoints(relDist) # get number of gauss points depending on relative distance
    GP,w = np.polynomial.legendre.leggauss(numGP) # get gauss points and weights on standard interval -1,1

    # initialize
    g_el = np.zeros((A['numNodPerEl'],1),dtype=complex)
    h_el = np.zeros((A['numNodPerEl'],1),dtype=complex)
    c_el = 0.

    for m in range(numGP): # gauss point loop m
        for n in range(numGP): # gauss point loop n
            
            eta = np.array([GP[n],GP[m]])
            weight1 = w[n]
            weight2 = w[m]
            shapeFunc = NiX(A['elTypesNum'][0], eta) # linear lagrangian polynomial 
            GPcoord = np.matmul(shapeFunc.T,elCoord) # discontinuous nodal coordinates of the mapped element

            # determinante
            shapeFuncDeriv = NigradX(A['elTypesNum'][0], eta)
            J = np.matmul(shapeFuncDeriv.T,elCoord)
            tvec1 = J[0, :] # tangential vector in cont nodes
            tvec2 = J[1, :]
            nvec = np.cross(tvec1,tvec2) # normal vector in cont nodes
            Jdet = np.linalg.norm(nvec)            
            nvec_unit = nvec/Jdet*conf['IntExt'] # unit normal vector

            # Green's function (fundamental solution)
            r_vec = GPcoord[0]-cpCoord
            r = np.linalg.norm(r_vec)
            r_gradnGP = np.dot(r_vec/r,nvec_unit)
            G = 1/(4*np.pi)*np.exp(1j*conf['k']*r)/r
            G_gradnGP = G*(1j*conf['k']-1/r)*r_gradnGP

            # interpolation function       
            IntFunc = NiXDiscSurf(B['numNodPerEl'],conf['DiscAlpha'],conf['DiscIntOrder'],eta)

            # element matrices
            g_el += G*IntFunc*Jdet*weight1*weight2
            h_el += G_gradnGP*IntFunc*Jdet*weight1*weight2
            
            # integral free term determined indirectly through quasi static problem
            c_el += 1/(4*np.pi*r**2)*r_gradnGP*Jdet*weight1*weight2

    return g_el,h_el,c_el

def singularIntegration(A,B,conf,cp,el,DOF,etaCP): 
    """
    Singular integration by using polar transformations. 
    """
    elindices = A['elements'][el,:] # cont node ids of the current element
    elCoord = A['nodes'][elindices, :] # cont nodal coordinates of the current element
    cpCoord = B['nodes'][cp,:] # cp coordinates
    numGP = 20 # set number of GP points to 20 for singular inegration
    GP,w = np.polynomial.legendre.leggauss(numGP) # get gauss points and weights on standard interval -1,1
    
    # initialize
    g_el = np.zeros((A['numNodPerEl'],1),dtype=complex)
    h_el = np.zeros((A['numNodPerEl'],1),dtype=complex)
    c_el = 0.
    #  number triangles on (quadrilaterial) element
    numTri = 4  
    # : adapt the following code and implement loop structure
    for i in range(numTri): # loop over triangles

        for m in range(numGP): # gauss point loop m
            for n in range(numGP): # gauss point loop n

                etaGP = np.array([GP[n],GP[m]])
                weight1 = w[n]
                weight2 = w[m]

                #  function call for polar transformation on triangle elements
                etaTri, JdetTri = polarTrafoTri(etaCP[DOF,:],etaGP,i)

                shapeFunc = NiX(A['elTypesNum'][0], etaTri) # linear lagrangian polynomial 
                GPcoord = np.matmul(shapeFunc.T,elCoord) # discontinuous nodal coordinates of the mapped element

                # determinante
                shapeFuncDeriv = NigradX(A['elTypesNum'][0], etaTri)
                J = np.matmul(shapeFuncDeriv.T,elCoord)
                tvec1 = J[0, :] # tangential vector in cont nodes
                tvec2 = J[1, :]
                nvec = np.cross(tvec1,tvec2) # normal vector in cont nodes
                Jdet = np.linalg.norm(nvec)
                nvec_unit = nvec/Jdet*conf['IntExt'] # unit normal vector

                # Green's function (fundamental solution)
                r_vec = GPcoord-cpCoord
                r = np.linalg.norm(r_vec)
                r_gradnGP = np.dot(r_vec/r,nvec_unit)
                G = 1/(4*np.pi)*np.exp(1j*conf['k']*r)/r
                G_gradnGP = G*(1j*conf['k']-1/r)*r_gradnGP

                # interpolation function       
                IntFunc = NiXDiscSurf(B['numNodPerEl'],conf['DiscAlpha'],conf['DiscIntOrder'],etaTri)

                # element matrices
                g_el += G*IntFunc*Jdet*JdetTri*weight1*weight2
                h_el += G_gradnGP*IntFunc*Jdet*JdetTri*weight1*weight2

                # integral free term determined indirectly through quasi static problem
                c_el += 1/(4*np.pi*r**2)*r_gradnGP*Jdet*JdetTri*weight1*weight2

    return g_el,h_el,c_el