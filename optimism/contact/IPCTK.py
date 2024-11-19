from .. import QuadratureRule
from functools import partial
import equinox as eqx
import ipctk
import jax
import jax.numpy as jnp
import meshio


class IPCTKContact(eqx.Module):
    mesh: any
    collision_mesh: any
    dof_manager: any
    max_neighbors: int
    q_rule: QuadratureRule.QuadratureRule
    dhat: float
    potential: ipctk.BarrierPotential

    def __init__(self, mesh_file, dof_manager, max_neighbors=3, q_degree=3, dhat=2e-3):
        self.mesh = meshio.read(mesh_file)
        self.max_neighbors = max_neighbors
        self.q_rule = QuadratureRule.create_quadrature_rule_1D(q_degree)

        rest_positions = self.mesh.points
        faces = self.mesh.cells_dict['triangle']
        edges = ipctk.edges(faces)
        self.dhat = dhat
        self.collision_mesh = ipctk.CollisionMesh(rest_positions, edges)
        self.potential = ipctk.BarrierPotential(self.dhat)
        self.dof_manager = dof_manager

    def collisions(self, coords):
        collisions = ipctk.Collisions()
        collisions.build(self.collision_mesh, coords, self.dhat) # performs culling to find only potential collisions with distances less than dhat
        return collisions

    def energy(self, Uu, p):
        return _energy(self, Uu, p)
    
    def gradient(self, Uu, p):
        return _gradient(self, Uu, p)

    def hvp(self, Uu, p, v):
        return jax.jvp(self.gradient, (Uu, p), (v, p))[1]

# annoying below since jvps don't really play nice with classes that well

@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _energy(contact, Uu, p):
    U = contact.dof_manager.create_field(Uu, p[0])
    coords = contact.collision_mesh.rest_positions.copy()[:, 0:2]
    curr_coords = U + coords
    collisions = contact.collisions(curr_coords)
    barrier_energy = contact.potential(collisions, contact.collision_mesh, curr_coords)
    return barrier_energy

@_energy.defjvp
def _jvp(contact, primals, tangents):
    Uu, p = primals
    dUu, dp = tangents
    U = contact.dof_manager.create_field(Uu, p[0])
    coords = contact.collision_mesh.rest_positions.copy()[:, 0:2]
    curr_coords = U + coords
    collisions = contact.collisions(curr_coords)
    barrier_energy = contact.potential(collisions, contact.collision_mesh, curr_coords)
    barrier_grad = contact.potential.gradient(collisions, contact.collision_mesh, curr_coords)
    return barrier_energy, jnp.dot(barrier_grad, dUu)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _gradient(contact, Uu, p):
    return jax.grad(contact.energy, argnums=0)(Uu, p)


@_gradient.defjvp
def _hvp(contact, primals, tangents):
    Uu, p = primals
    dUu, dp = tangents
    U = contact.dof_manager.create_field(Uu, p[0])
    coords = contact.collision_mesh.rest_positions.copy()[:, 0:2]
    curr_coords = U + coords
    collisions = contact.collisions(curr_coords)
    barrier_grad = contact.potential.gradient(collisions, contact.collision_mesh, curr_coords)
    barrier_hess = contact.potential.hessian(collisions, contact.collision_mesh, curr_coords)
    return barrier_grad, barrier_hess @ dUu
