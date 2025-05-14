import jax
import jax.numpy as np
import numpy as onp

import unittest
from . import MeshFixture

from optimism import FunctionSpace
from optimism.material import LinearElastic as MatModel
#from optimism.material import Neohookean as MatModel
from optimism import Mesh
from optimism import Mechanics
from optimism.EquationSolver import newton_solve
from optimism import QuadratureRule
from optimism.TensorMath import tensor_2D_to_3D

import sys
haveMetis = 'metis' in sys.modules
skipMessage = 'metis not installed, PatchTestQuadraticElements disabled'
if(haveMetis):
    import metis

E = 1.0
nu = 0.3
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'strain measure': 'linear'}


def insort(a, b, kind='mergesort'):
    c = onp.concatenate((a, b))
    c.sort(kind=kind)
    flag = onp.ones(len(c), dtype=bool)
    onp.not_equal(c[1:], c[:-1], out=flag[1:])
    return c[flag]


def create_graph(conns):
    
    elemToElem = [ [] for _ in range(len(conns)) ]
    _, edges = Mesh.create_edges(conns)
    for edge in edges:
        t0 = edge[0]
        t1 = edge[2]
        if t1 > -1:
            elemToElem[t0].append(t1)
            elemToElem[t1].append(t0)

    #return elemToElem

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
    (edgecuts, parts) = metis.part_graph(graph, 4)
    return np.array(parts, dtype=int)


def construct_basis_on_poly(elems, conns, fs : FunctionSpace.FunctionSpace):
    uniqueNodes = conns[elems[0]]
    for e in elems[1:]:
        uniqueNodes = insort(uniqueNodes, conns[e])

    # these are not set to be general yet, assuming the pou integration is over all nodes, all nodes are active
    Q = len(uniqueNodes) # number of quadrature bases
    Nnode = len(uniqueNodes) # number of active nodes on poly

    globalNodeToLocalNode = {nodeN : n for n, nodeN in enumerate(uniqueNodes)}

    G = onp.zeros((Q,Q))
    for e in elems:
        elemQuadratureShapes = fs.shapes[e]
        vols = fs.vols[e]
        for n, node in enumerate(conns[e]):
            node = int(node)
            for m, mode in enumerate(conns[e]):
                mode = int(mode)
                N = globalNodeToLocalNode[node]
                M = globalNodeToLocalNode[mode]
                G[N,M] += vols @ (elemQuadratureShapes[:,n] * elemQuadratureShapes[:,m])

    S,U = onp.linalg.eigh(G)
    nonzeroS = abs(S) > 1e-14
    Sinv = 1.0/S[nonzeroS]
    Uu = U[:, nonzeroS]

    Ginv = Uu@onp.diag(Sinv)@Uu.T

    dim = fs.shapeGrads.shape[3]
    B = onp.zeros((Q, Nnode, fs.shapeGrads.shape[3]))
    for e in elems:
        elemQuadratureShapes = fs.shapes[e]
        elemShapeGrads = fs.shapeGrads[e]

        vols = fs.vols[e]
        for n, node in enumerate(conns[e]):
            node = int(node)
            for m, mode in enumerate(conns[e]):
                mode = int(mode)
                N = globalNodeToLocalNode[node]
                M = globalNodeToLocalNode[mode]
                for d in range(dim):
                    B[N,M,d] += vols @ (elemQuadratureShapes[:,n] * elemShapeGrads[:,m,d])

    for n in range(Nnode):
        for d in range(dim):
            B[:,n,d] = Ginv@B[:,n,d]

    # attemps to diagonalize the Graham matrix
    #G1 = onp.average(G, axis=1)
    #GinvG1 = Ginv @ G1

    #V = onp.outer( onp.ones(Nnode), GinvG1 )
    #Gtilde = V @ G @ V.T

    #print('Gtilesum = ', onp.sum(onp.sum(Gtilde,axis=0),axis=0))
    #print('Gsum = ', onp.sum(onp.sum(G,axis=0),axis=0))
    #print('Gtilde = ', Gtilde)

    return B, np.sum(G, axis=1), globalNodeToLocalNode


class PatchTestQuadraticElements(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 5
        self.Ny = 5

        xRange = [0.,1.]
        yRange = [0.,1.]
        self.mesh, _ = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange, lambda x : x)
        self.partition = create_partitions(self.mesh.conns) # partitioning seems to only work on lower order mesh at the moment
        #self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=2, createNodeSetsFromSideSets=True)

        self.quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, self.quadRule)
        self.materialModel = MatModel.create_material_model_functions(props)


    def create_polys(self):
        polys = {}
        for e,p in enumerate(self.partition):
            p = int(p)
            e = int(e)
            if p in polys:
                polys[p].append(e)
            else:
                polys[p] = [e]
        return polys


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
    

    @unittest.skipIf(not haveMetis, skipMessage)
    def test_dirichlet_patch_test_with_quadratic_elements(self):
        ebcs = [FunctionSpace.EssentialBC(nodeSet='all_boundary', component=0),
                FunctionSpace.EssentialBC(nodeSet='all_boundary', component=1)]
        self.dofManager = FunctionSpace.DofManager(self.fs, self.mesh.coords.shape[1], ebcs)

        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]]) 
        self.UTarget = self.mesh.coords@self.targetDispGrad.T
        Ubc = self.dofManager.get_bc_values(self.UTarget)

        polys = self.create_polys()
        polyFs = []

        totalVol = 0.0
        for p in polys:
            pElems = polys[p]
            B,W,globalToLocalNode = construct_basis_on_poly(pElems, self.mesh.conns, self.fs)
            polyFs.append((B,W,globalToLocalNode))
            pvol = np.sum(W)
            totalVol += pvol
            gradUs = np.zeros((B.shape[0],B.shape[2],B.shape[2]))
            for node in globalToLocalNode:
                n = globalToLocalNode[node]
                b = B[:,n,:]
                UatN = self.UTarget[node]
                gradUs += jax.vmap(lambda bb : np.outer(UatN,bb))(b)

            for gradU in gradUs:
                self.assertArrayNear(gradU, self.targetDispGrad, 8)   

        def objective_poly(Uu):

            def energy_density(gradu):
                g = tensor_2D_to_3D(gradu)
                return self.materialModel.compute_energy_density(g, None, 0.0)

            U = self.dofManager.create_field(Uu, Ubc)
            totalEnergy = 0.0
            for B,W,gToLMap in polyFs:
                gradU = np.zeros((B.shape[0],B.shape[2],B.shape[2]))
                for node in gToLMap:
                    n = gToLMap[node]
                    b = B[:,n,:]
                    UatN = U[node]
                    gradU += jax.vmap(lambda bb : np.outer(UatN,bb))(b)

                energies = jax.vmap(energy_density)(gradU)
                totalEnergy += W @ energies

            return totalEnergy

        Uu, solverSuccess = newton_solve(objective_poly, self.dofManager.get_unknown_values(0.0*self.UTarget))
        self.assertTrue(solverSuccess)

        self.write_output(Uu, Ubc, 0)

        U = self.dofManager.create_field(Uu, Ubc)
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, U)
        ne, nqpe = self.fs.vols.shape
        for dg in dispGrads.reshape(ne*nqpe,2,2):
            self.assertArrayNear(dg, self.targetDispGrad, 14)

        grad_func = jax.grad(objective_poly)
        self.assertNear(np.linalg.norm(grad_func(Uu)), 0.0, 14)


if __name__ == '__main__':
<<<<<<< HEAD:test/test_PatchTestPou.py
    unittest.main()
=======
    unittest.main()
>>>>>>> main:optimism/test/test_PatchTestPou.py
