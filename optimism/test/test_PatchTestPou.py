import jax
import jax.numpy as np
import numpy as onp
import scipy as scipy

from optimism import FunctionSpace
from optimism.material import Neohookean as MatModel
from optimism import Mesh
from optimism import Mechanics
from optimism.Timer import Timer
from optimism.EquationSolver import newton_solve
from optimism import QuadratureRule
from optimism.test import MeshFixture
import metis

E = 1.0
nu = 0.3
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'strain measure': 'linear'}


def create_graph(conns):

    def insort(a, b, kind='mergesort'):
        # took mergesort as it seemed a tiny bit faster for my sorted large array try.
        c = onp.concatenate((a, b)) # we still need to do this unfortunatly.
        c.sort(kind=kind)
        flag = onp.ones(len(c), dtype=bool)
        onp.not_equal(c[1:], c[:-1], out=flag[1:])
        return c[flag]

    nodeToElem = {}
    for e, elem in enumerate(conns):
        for n in elem:
            n = int(n)
            if n in nodeToElem:
                nodeToElem[n].append(e)
            else:
                nodeToElem[n] = [e]

    elemToElem = [None] * len(conns)
    for e, elem in enumerate(conns):
        elemToElem[e] = nodeToElem[int(elem[0])]
        for n in elem[1:]:
            n = int(n)
            elemToElem[e] = insort(elemToElem[e], nodeToElem[n])
        elemToElem[e] = list(elemToElem[e])

    return elemToElem


def create_partitions(conns):
    graph = create_graph(conns)
    (edgecuts, parts) = metis.part_graph(graph, 8)
    return np.array(parts, dtype=int)


def construct_basis_on_poly(elems, conns, fs : FunctionSpace.FunctionSpace):
    nodalShapes = {}
    for e in elems:
        elemShapes = fs.shapes[e]

        for n, node in enumerate(conns[e]):
            node = int(node)
            if node in nodalShapes:
                nodalShapes[node][e] = elemShapes[:,n]
            else:
                nodalShapes[node] = {e : elemShapes[:,n]}

    Q = len(nodalShapes)
    G = onp.zeros((Q,Q))

    for n, nodeN in enumerate(nodalShapes):
        elemsAndWeightsN = nodalShapes[nodeN]
        for m, nodeM in enumerate(nodalShapes):
            elemsAndWeightsM = nodalShapes[nodeM]
            for elemN in elemsAndWeightsN:
                if elemN in elemsAndWeightsM:
                    integralOfShapes = fs.vols[elemN] @ (elemsAndWeightsN[elemN] * elemsAndWeightsM[elemN])
                    G[n,m] = integralOfShapes
                    #G = G.at[n,m].set(integralOfShapes)

    S,U = onp.linalg.eigh(G)
    print('diff = ', U@np.diag(S)@U.T - G)


    
    return nodalShapes


class PatchTestQuadraticElements(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 5
        self.Ny = 5

        xRange = [0.,1.]
        yRange = [0.,1.]
        self.mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : x)
        # self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(mesh, order=2, createNodeSetsFromSideSets=True)

        self.partition = create_partitions(self.mesh.conns)

        polys = {}
        for e,p in enumerate(self.partition):
            p = int(p)
            e = int(e)
            if p in polys:
                polys[p].append(e)
            else:
                polys[p] = [e]

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)


        for p in polys:
            construct_basis_on_poly(polys[p], self.mesh.conns, self.fs)


        materialModel = MatModel.create_material_model_functions(props)

        mcxFuncs = \
            Mechanics.create_mechanics_functions(self.fs,
                                                 "plane strain",
                                                 materialModel)

        self.compute_energy = mcxFuncs.compute_strain_energy
        self.internals = mcxFuncs.compute_initial_state()


    def write_output(self, Uu, Ubc, step):
        from optimism import VTKWriter
        U = self.dofManager.create_field(Uu, Ubc)
        plotName = 'patch-'+str(step).zfill(3)
        writer = VTKWriter.VTKWriter(self.mesh, baseFileName=plotName)

        writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)

        bcs = np.array(self.dofManager.isBc, dtype=int)
        writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

        writer.add_cell_field(name='partition',
                              cellData=self.partition,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)

        writer.write()
    

    def test_dirichlet_patch_test_with_quadratic_elements(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]
        self.dofManager = FunctionSpace.DofManager(self.fs, self.mesh.coords.shape[1], ebcs)

        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        self.UTarget = self.mesh.coords@self.targetDispGrad.T
        Ubc = self.dofManager.get_bc_values(self.UTarget)
        
        @jax.jit
        def objective(Uu):
            U = self.dofManager.create_field(Uu, Ubc)
            return self.compute_energy(U, self.internals)
        
        #with Timer(name="NewtonSolve"):
        Uu = newton_solve(objective, self.dofManager.get_unknown_values(self.UTarget))

        self.write_output(Uu, Ubc, 0)

        U = self.dofManager.create_field(Uu, Ubc)
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, U)
        ne, nqpe = self.fs.vols.shape
        for dg in dispGrads.reshape(ne*nqpe,2,2):
            self.assertArrayNear(dg, self.targetDispGrad, 14)

        grad_func = jax.grad(objective)
        self.assertNear(np.linalg.norm(grad_func(Uu)), 0.0, 14)


if __name__ == '__main__':
    import unittest
    unittest.main()