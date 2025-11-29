from common_modules import np

def NiX(eltype, xiEta): 
    """
    get shape function
    """
    xi,eta = [*xiEta[:]]
    eltypes ={
        # 2D
        3:np.array([[(eta/2 - 1/2)*(xi/2 - 1/2)],
                    [-(eta/2 - 1/2)*(xi/2 + 1/2)],
                    [(eta/2 + 1/2)*(xi/2 + 1/2)],
                    [-(eta/2 + 1/2)*(xi/2 - 1/2)]]) # 4-node quadrangle
       }
    return eltypes.get(eltype, 'Not supported type') 

def NigradX(eltype, xiEta): 
    """
    get shape function derivative
    """
    xi,eta = [*xiEta[:]]
    eltypes ={
        # 2D
        3:np.array([[eta/4 - 1./4,   xi/4 - 1./4],
                    [1./4 - eta/4, - xi/4 - 1./4],
                    [eta/4 + 1./4,   xi/4 + 1./4],
                    [- eta/4 - 1./4, 1./4 - xi/4]]) # 4-node quadrangle
       }
    return eltypes.get(eltype, 'Not supported type') 


def NiXDiscSurf(nodesPerEl,alpha,order,xiEta): 
    """
    get local coordinates of the mapped element
    """
    assert order == 1 and nodesPerEl == 4 , "Not supported order or element type"
    xi,eta = [*xiEta[:]]
    zeta = 1 - alpha
    if nodesPerEl == 4:
        if order == 1: # linear polynomials, 4-node quadrilateral element
            Ni = [[(1/2 - xi/(2*zeta))*(1/2 - eta/(2*zeta))],
                          [(1/2 + xi/(2*zeta))*(1/2 - eta/(2*zeta))],
                          [(1/2 + xi/(2*zeta))*(1/2 + eta/(2*zeta))],
                          [(1/2 - xi/(2*zeta))*(1/2 + eta/(2*zeta))]]
    return np.array(Ni)