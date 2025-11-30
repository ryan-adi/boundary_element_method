from common_modules import os, nb, Dict, unicode_type, float64
from src.read_config import Configuration
from src.local_coordinates import evalLocCoord
from src.mesh import Mesh
from src.processing import Processing
from src.visualize import *
from src.utils import export_csv

#@nb.njit
def make_nb_dict(dict):
    d = Dict.empty(key_type=unicode_type, value_type=float64)
    for key, val in dict.items():
        d[key] = val
    return d

if __name__=="__main__":
    # paths
    cwd = os.getcwd()

    # read configuration files for simulation parameters
    xml_path = os.path.join(cwd, "config.xml")
    configuration = Configuration()
    configuration.read(xml_path)
    conf = configuration.get()
    # conf_nb = make_nb_dict(conf)

    # read mesh file
    testcase = os.path.join("msh", "ductQuadMesh.msh")
    mesh = Mesh()
    geoA = mesh.read_mesh(testcase, conf['IntExt']) 
    # map to parameter space, convert to discontinuous mesh ###
    eta = evalLocCoord(conf['DiscAlpha'],conf['DiscElementOrder']) 
    geoB = mesh.discretize_mesh(geoA,eta) 

    # visualize mesh
    # contMeshPlot(geoA,True,True,True,240,0.07)
    # discMeshPlot(geoA,geoB,False,True,True,240,0.14)
    
    # solve linear equations
    processing = Processing(conf, geoA, eta, geoB)
    processing.assemble_matrix()
    processing.boundary_conditions()
    p = processing.solve()
    p_data = {"p_real": np.squeeze(p[:,:].real), 
              "p_imag": np.squeeze(p[:,:].imag)}
    csv_path = os.path.join(cwd, "pressure.csv")
    export_csv(csv_path, p_data)

    ## ===== POSTPROCESSING ===== ##
    plot_2d(geoB, p)