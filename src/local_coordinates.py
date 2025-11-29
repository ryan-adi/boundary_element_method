from common_modules import np

def getLocalCoordinates(eltype): 
    """
    get local coordinates of nodes of continous mesh
    """
    eltypes ={
        # 2D
        3:np.array([[-1,-10],[1,-1],[1,1],[-1,1]]) # 4-node quadrangle
    }
    return eltypes.get(eltype, 'Not supported type')

def evalLocCoord(alpha,order): 
    """
    evaluate the local coordinates of the current collocation point
    """
    assert order == 1, "Not supported order"
    if order == 1: 
        localCoord = [-1+alpha, 1-alpha]
        eta = [localCoord[0],localCoord[0]],[localCoord[1],localCoord[0]],[localCoord[1],localCoord[1]],[localCoord[0],localCoord[1]]        
    return np.array(eta)