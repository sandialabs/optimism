# fea data structures
from optimism.Mesh import *
from optimism import Surface
from optimism import QuadratureRule

# solver
from optimism.EquationSolver import newton_solve

# timing utils
from optimism.Timer import Timer

# testing utils
from optimism.test.TestFixture import *

d_kappa = 1.0
d_nu = 0.3
d_E = 3*d_kappa*(1 - 2*d_nu)
defaultProps = {'elastic modulus': d_E,
                'poisson ratio': d_nu}


def compute_residual_norm(grad_func, dofValues):
    grad = grad_func(dofValues)
    return np.linalg.norm(grad)

def map_to_arch(x, R):
    r = R-x[0]
    y = x[1]
    pi = np.pi
    return r*np.array([-np.cos(pi*y), np.sin(pi*y)])

def map_to_cos(x, L, warp):
    y = x[1]
    pi = np.pi
    return np.array([L*(y-0.5), warp*np.cos(2*pi*(y-0.5))-x[0]])

class MeshFixture(TestFixture):
    
    def create_mesh_and_disp(self, Nx, Ny, xRange, yRange, initial_disp_func, setNamePostFix=''):
        coords, conns = create_structured_mesh_data(Nx, Ny, xRange, yRange)
        tol = 1e-8
        nodeSets = {}
        nodeSets['left'+setNamePostFix] = np.flatnonzero(coords[:,0] < xRange[0] + tol)
        nodeSets['bottom'+setNamePostFix] = np.flatnonzero(coords[:,1] < yRange[0] + tol)
        nodeSets['right'+setNamePostFix] = np.flatnonzero(coords[:,0] > xRange[1] - tol)
        nodeSets['top'+setNamePostFix] = np.flatnonzero(coords[:,1] > yRange[1] - tol)
        nodeSets['all_boundary'+setNamePostFix] = np.flatnonzero((coords[:,0] < xRange[0] + tol) |
                                            (coords[:,1] < yRange[0] + tol) |
                                            (coords[:,0] > xRange[1] - tol) |
                                            (coords[:,1] > yRange[1] - tol) )
        
        def is_edge_on_left(xyOnEdge):
            return np.all( xyOnEdge[:,0] < xRange[0] + tol  )

        def is_edge_on_bottom(xyOnEdge):
            return np.all( xyOnEdge[:,1] < yRange[0] + tol  )

        def is_edge_on_right(xyOnEdge):
            return np.all( xyOnEdge[:,0] > xRange[1] - tol  )
        
        def is_edge_on_top(xyOnEdge):
            return np.all( xyOnEdge[:,1] > yRange[1] - tol  )

        sideSets = {}
        sideSets['left'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_left)
        sideSets['bottom'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_bottom)
        sideSets['right'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_right)
        sideSets['top'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_top)
        
        allBoundaryEdges = np.vstack([s for s in sideSets.values()])
        sideSets['all_boundary'+setNamePostFix] = allBoundaryEdges

        blocks = {'block'+setNamePostFix: np.arange(conns.shape[0])}
        mesh = construct_mesh_from_basic_data(coords, conns, blocks, nodeSets, sideSets)
        return mesh, vmap(initial_disp_func)(mesh.coords)


    def create_arch_mesh_disp_and_edges(self, N, M, w, R,
                                        bcSetFraction=1./3.,
                                        setNamePostFix=''):
        h = 1.0
        coords, conns = create_structured_mesh_data(N, M, [-w, w], [0., h])

        tol = 1e-8
        nodeSets = {}
        nodeSets['left'+setNamePostFix] = np.flatnonzero(coords[:,1] < 0.0 + tol)
        nodeSets['right'+setNamePostFix] = np.flatnonzero(coords[:,1] > h - tol)
        nodeSets['bottom'+setNamePostFix] = np.flatnonzero(coords[:,0] > w - tol)
        nodeSets['top'+setNamePostFix] = np.flatnonzero(coords[:,0] < -w + tol)
        nodeSets['top_left'+setNamePostFix] = np.intersect1d(nodeSets['top'+setNamePostFix],
                                                             nodeSets['left'+setNamePostFix])
        a = 0.5*h*(1.0 - bcSetFraction)
        b = 0.5*h*(1.0 + bcSetFraction)
        
        nodeSets['push'+setNamePostFix] = np.flatnonzero( (coords[:,0] < tol - w)
                                                          & (coords[:,1] > a)
                                                          & (coords[:,1] < b) )

        def is_edge_on_left(xyOnEdge):
            return np.all(xyOnEdge[:,1] < 0.0 + tol)
        
        def is_edge_on_bottom(xyOnEdge):
            return np.all( xyOnEdge[:,0] > w - tol  )

        def is_edge_on_right(xyOnEdge):
            return np.all( xyOnEdge[:,1] > h - tol  )
        
        def is_edge_on_top(xyOnEdge):
            return np.all( xyOnEdge[:,0] < tol - w  )

        def is_edge_on_loading_patch(xyOnEdge):
            return np.all((xyOnEdge[:,0] < tol - w) & (xyOnEdge[:,1] > a) & (xyOnEdge[:,1] < b))

        sideSets = {}
        sideSets['top'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_top)
        sideSets['bottom'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_bottom)
        sideSets['left'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_left)
        sideSets['right'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_right)
        sideSets['push'+setNamePostFix] = Surface.create_edges(coords, conns, is_edge_on_loading_patch)
        
        coords = vmap(map_to_arch, (0,None))(coords, R)
        blocks = {'block'+setNamePostFix: np.arange(conns.shape[0])}
        U = np.zeros(coords.shape)
        return construct_mesh_from_basic_data(coords, conns, blocks, nodeSets, sideSets), U

    
    def create_cos_mesh_disp_and_edges(self, N, M, w, L, warp):
        h = 1.0
        coords, conns = create_structured_mesh_data(N, M, [-w, w], [0., h])
        
        tol = 1e-8
        nodeSets = {}
        nodeSets['left'] = np.flatnonzero(coords[:,1] < 0.0 + tol)
        nodeSets['right'] = np.flatnonzero(coords[:,1] > h - tol)
        nodeSets['top'] = np.flatnonzero( (coords[:,0] < tol - w)
                                          & (coords[:,1] > 0.3)
                                          & (coords[:,1] < 0.7) )

        def is_edge_on_left(xyOnEdge):
            return np.all( xyOnEdge[:,0] < -w + tol  )

        sideSets = {}
        sideSets['top'] = Surface.create_edges(coords, conns, is_edge_on_left)
        
        coords = vmap(map_to_cos, (0,None,None))(coords, L, warp)
        blocks = {'block': np.arange(conns.shape[0])}
        U = np.zeros(coords.shape)
        return construct_mesh_from_basic_data(coords, conns, None, nodeSets, sideSets), U
    
