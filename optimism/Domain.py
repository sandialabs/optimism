from optimism import SparseMatrixAssembler
from optimism.FunctionSpace import DofManager, EssentialBC
from optimism.FunctionSpace import construct_function_space
from optimism.Mechanics import MechanicsFunctions
from optimism.Mechanics import create_multi_block_mechanics_functions
from optimism.Mesh import Mesh
from optimism.Objective import param_index_update
from optimism.QuadratureRule import create_quadrature_rule_1D, create_quadrature_rule_on_triangle
from typing import List, Optional
import equinox as eqx
import jax.numpy as np


class Domain(eqx.Module):
  mesh: Mesh
  ebcs: List[EssentialBC]
  disp_nset: str # TODO remove eventually
  mech_funcs: MechanicsFunctions
  dof: DofManager

  def __init__(
    self, mesh, ebcs, mat_models, mode2D, 
    disp_nset: Optional[str] = None, # TODO eventually deprecate this in favor of having lambdas bound to EssentialBCs
    p_order: Optional[int] = 1, # TODO hook this up
    q_order: Optional[int] = 2
  ):
    self.mesh = mesh
    self.ebcs = ebcs
    self.disp_nset = disp_nset # TODO eventually deprecate this
    q_rule_1d = create_quadrature_rule_1D(q_order)
    q_rule_2d = create_quadrature_rule_on_triangle(q_order)
    fspace = construct_function_space(mesh, q_rule_2d)
    # self.mech_funcs = create_mechanics_functions(fspace, mode2D, mat_model)
    self.mech_funcs = create_multi_block_mechanics_functions(fspace, mode2D, mat_models)
    self.dof = DofManager(fspace, 2, ebcs)

  def assemble_sparse(self, Uu, p):
    U = self.update_field(Uu, p)
    state = p[1]
    element_stiffnesses = self.mech_funcs.compute_element_stiffnesses(U, state)
    fspace = self.mech_funcs.fspace
    return SparseMatrixAssembler.\
        assemble_sparse_stiffness_matrix(element_stiffnesses, fspace.mesh.conns, self.dof)

  def create_field(self, Uu, Ubc):
    return self.dof.create_field(Uu, Ubc)

  def create_unknowns(self):
    Uu = self.dof.get_unknown_values(np.zeros(self.mesh.coords.shape))
    return Uu

  def energy_function(self, Uu, p):
    U = self.update_field(Uu, p)
    state = p[1]
    return self.mech_funcs.compute_strain_energy(U, state)

  def get_bc_values(self, U):
    return self.dof.get_bc_values(U)

  def get_ubcs(self, p):
    t = p[4] # use time
    V = np.zeros(self.mesh.coords.shape)
    for bc in self.ebcs:
      index = (self.mesh.nodeSets[bc.nodeSet], bc.component)
      val = bc.func(t)
      V = V.at[index].set(val)
    return self.dof.get_bc_values(V)

  def get_unknown_values(self, U):
    return self.dof.get_unknown_values(U)

  def update_bcs(self, p):
    return param_index_update(p, 0, self.get_ubcs(p))

  def update_field(self, Uu, p):
    Ubc = self.get_ubcs(p)
    return self.create_field(Uu, Ubc)

  def update_time(self, p, dt):
    t = p[4]
    return param_index_update(p, 4, t + dt)
