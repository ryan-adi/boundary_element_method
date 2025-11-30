from common_modules import np,  meshio
from .local_coordinates import getLocalCoordinates
from .shape_function import NiX, NigradX

class Mesh:
    def __init__(self):
        self.geoA = 0
        self.geoB = 0

    def read_mesh(self, file, intExt): 
        """
        Read physical mesh from GMSH
        characterize mesh properties
        transfer to isoparametric space
        calculate Jacobian and normalized normals
        """
        
        # read mesh file, extract necessary information
        mesh = meshio.read(file,)   
        elTypesName = [*mesh.cells_dict.keys()] # element types in mesh file | alt: [key for key in mesh.cells_dict.keys()]
        if len(elTypesName) > 1: "Warning: More than one element type in mesh file. Mesh properties are not unique and cannot be handeled!"
        nodes = mesh.points # node coordinates
        elements = np.array(*mesh.cells_dict.values()) # node IDs on elements
        numElements = elements.shape[0] # total number of elements
        numNodPerEl = elements.shape[1] # number of nodes per element
        numNodes = mesh.points.shape[0] # total number of nodes   

        ## CENTERPOINTS AND ELEMENT NORMAL VECTORS ##

        elnormalvector = np.zeros((numElements,3),dtype=float)
        elcenterpoint = np.zeros((numElements,3),dtype=float)

        for el in range(numElements):
            # Find the position and the normal vector at the center of the element        
            elcenterpoint[el,:] = np.mean(nodes[elements[el,:],:], axis=0)
            tmpVectora = nodes[elements[el,0],:] - nodes[elements[el,1],:] #Diff coord nodes 1-2
            tmpVectorb = nodes[elements[el,2],:] - nodes[elements[el,1],:] #Diff coord nodes 3-2
            normalvector = np.cross(tmpVectorb, tmpVectora) # normal vec by cross product
            elnormalvector[el,:] = intExt * normalvector # vec direction given by interior/exterior problem

        # Normalize normal vectors
        tmp = np.sqrt(np.sum(elnormalvector**2,axis=1))
        elnormalvector = elnormalvector/tmp[:,None]

        ## NODAL NORMAL AND TANGENTIAL VECTORS ##
        
        # initalize normal and tangetial vec matrizes
        elnodalnormals = np.zeros((numNodPerEl, 3, numElements),dtype=float) # elementwise nodal normals 

        for el in range(numElements): # element loop
            
            # extract nodal coordinates
            elindices = elements[el,:] # node ids of the current element
            elCoord = nodes[elindices, :] # nodal coordinates of the current element

            # parameter space coordinates
            if elTypesName[0] == 'quad':elTypesNum = [3]
            elXiEta = getLocalCoordinates(elTypesNum[0])

            # calc Jacobian
            J = np.zeros((2, 3, numNodPerEl),dtype=float)
            for nd in range(numNodPerEl): # for each node
                J[:, :, nd] = np.matmul(NigradX(elTypesNum[0], elXiEta[nd, :]).T, elCoord)
                # Wu - Boundary Element Acoustics, p. 53-54
                tvec1 = J[0, :, nd] # tangential vector in cont nodes
                tvec2 = J[1, :, nd]
                nvec = np.cross(tvec1,tvec2) # normal vector in cont nodes
                Jdet = np.linalg.norm(nvec)
                nvec = nvec/Jdet # unit normal vector
                # assemble
                elnodalnormals[nd, :, el] = intExt * nvec
            
        self.geoA = {'nodes':nodes,
                'elements':elements,
                'numNodes':numNodes,
                'numElements':numElements,
                'numNodPerEl':numNodPerEl,
                'elTypesName':elTypesName,
                'elTypesNum':elTypesNum,
                'elcenterpoint':elcenterpoint, 
                'elnormalvector':elnormalvector,
                'elnodalnvec':elnodalnormals        
            }
        
        return self.geoA
    
    def discretize_mesh(self, A, eta): 
        """
        map in isoparametric space
        transfer continuous Lagrangian elements into discontinuous elements (no shared nodes)
        """

        # initialize
        numNodes = A['numElements']*A['numNodPerEl'] # number discontinuous nodes
        numElements = A['numElements'] # number elements (doesn't change for disc mesh)
        numNodPerEl = A['numNodPerEl'] # number nodes per elements (doesn't change for disc mesh)
        nodes = np.zeros((numNodes, 3),dtype=float) 
        elements = np.zeros((numElements, numNodPerEl),dtype=float)
        nodalnvec = np.zeros((numNodes,3),dtype=float)

        counter = 0
        counter2 = 0
        for el in range(A['numElements']): # element loop

            # extract nodal information on physical continous mesh
            elindices = A['elements'][el,:] # cont node ids of the current element
            elCoord = A['nodes'][elindices, :] # cont nodal coordinates of the current element
            nvecCont = A['elnodalnvec'][:,:,el] # nodal normal vectors
            elements[el,:] = range(counter,counter + A['numNodPerEl']) # new sorted node ids for disc mesh

            counter += A['numNodPerEl']

            for DOF in range(A['numNodPerEl']): # node loop on every element

                shapeFunc = NiX(A['elTypesNum'][0], eta[DOF,:]) # linear lagrangian polynomial 
                discCoord = np.matmul(shapeFunc.T,elCoord) # discontinuous nodal coordinates of the mapped element

                # Get mapped local coordinate system in discontinuous nodes        
                nvec = np.matmul(shapeFunc.T,nvecCont) # discontinuous nodal normal vectors of the mapped element

                # fill matrices
                nodes[counter2,:] = discCoord
                nodalnvec[counter2,:] = nvec

                counter2 += 1
                                                                    
        self.geoB = {'nodes':nodes,
            'elements':elements,
            'numNodes':numNodes,
            'numElements':numElements,
            'numNodPerEl':numNodPerEl,
            'nodalnvec':nodalnvec        
        }
                                                                        
        return self.geoB
