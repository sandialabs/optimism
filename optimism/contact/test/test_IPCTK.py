import equinox as eqx
import ipctk
import jax
import jax.numpy as jnp
import meshio
import unittest
from optimism import VTKWriter
from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism.FunctionSpace import DofManager
from optimism.test.MeshFixture import MeshFixture
from optimism.contact import Contact, IPCTK
from optimism.contact.IPCTK import IPCTKContact


def write_vtk_mesh(mesh, meshName):
    writer = VTKWriter.VTKWriter(mesh, baseFileName=meshName)
    writer.write()


class TwoBodyICPTKContactFixture(MeshFixture):

    def setUp(self):
        self.targetDispGrad = np.array([[0.1, -0.2],[0.4, -0.1]])
        
        m1 = self.create_mesh_and_disp(3, 5, [0.0, 1.0], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '1')

        m2 = self.create_mesh_and_disp(2, 4, [1.001, 2.001], [0.0, 1.0],
                                        lambda x : self.targetDispGrad.dot(x), '2')
        
        self.mesh, _ = Mesh.combine_mesh(m1, m2)

        order=1 # ipctk only works with 1st order meshes
        self.mesh = Mesh.create_higher_order_mesh_from_simplex_mesh(self.mesh, order=order, copyNodeSets=False)

        nodeSets = Mesh.create_nodesets_from_sidesets(self.mesh)
        self.mesh = Mesh.mesh_with_nodesets(self.mesh, nodeSets)
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        self.dofManager = DofManager(self.fs, dim=self.mesh.coords.shape[1], EssentialBCs=[])

        write_vtk_mesh(self.mesh, 'mesh')

        self.contact = IPCTKContact('mesh.vtk', self.dofManager)
        self.disp = np.zeros(self.mesh.coords.shape)

    def test_collisions_min_distance(self):

        coords = self.contact.collision_mesh.rest_positions.copy() # just use rest positions
        collision_mesh = self.contact.collision_mesh
        # collisions = self.contact.collisions(coords)
        self.contact = self.contact.update_collisions(coords)
        collisions = self.contact.collisions

        # test distance computation in initial positions (disp=0)
        coords[:,0] += self.disp[:,0]
        coords[:,1] += self.disp[:,1]
        self.assertNear(np.sqrt(collisions.compute_minimum_distance(collision_mesh, coords)), 1e-3, 8)

        # test distance computation in offset positions
        index = (self.mesh.nodeSets['right1'], 0)
        self.disp = self.disp.at[index].set(5e-4)
        coords[:,0] += self.disp[:,0]
        coords[:,1] += self.disp[:,1]
        self.assertNear(np.sqrt(collisions.compute_minimum_distance(collision_mesh, coords)), 5e-4, 8)

    def test_energy(self):
        U = 0. * self.disp
        # Uu = self.dofManager.get_unknown_values(np.zeros(self.mesh.coords.shape))
        # p = Objective.Params()
        # barrier_energy = self.contact.energy(Uu, p)
        # barrier_energy = self.contact.energy(U)
        # self.assertTrue(barrier_energy > 0.0)
        # # barrier_grad = self.contact.gradient(Uu, p)
        # barrier_grad = self.contact.gradient(U)

        # U = self.contact.dof_manager.create_field(Uu, p[0])
        coords = self.contact.collision_mesh.rest_positions.copy()[:, 0:2]
        curr_coords = U + coords
        # collisions = self.contact.collisions(curr_coords)
        self.contact = self.contact.update_collisions(curr_coords)
        collisions = self.contact.collisions

        barrier_energy = self.contact.energy(U)
        self.assertTrue(barrier_energy > 0.0)
        barrier_grad = self.contact.gradient(U)
        # barrier_grad_jit = jit(self.contact.gradient)(U)
        true_grad = self.contact.potential.gradient(collisions, self.contact.collision_mesh, curr_coords)
        self.assertArrayNear(barrier_grad.flatten(), true_grad, 12)
        # self.assertArrayNear(barrier_grad_jit.flatten(), true_grad, 12)

        # barrier_grad_jit = eqx.filter_jit(self.contact.gradient)(U)
        barrier_grad_jit = eqx.filter_jit(IPCTK._gradient)(self.contact, U)
        # print(barrier_grad_jit)

        # v = jnp.ones(Uu.shape[0])
        # barrier_hvp = self.contact.hvp(Uu, p, v)
        # true_hess = self.contact.potential.hessian(collisions, self.contact.collision_mesh, curr_coords)
        # true_hvp = true_hess @ v
        # self.assertArrayNear(barrier_hvp, true_hvp, 12)

if __name__ == '__main__':
    unittest.main()
