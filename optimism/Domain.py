from optimism import SparseMatrixAssembler
from optimism.FunctionSpace import DofManager, EssentialBC
from optimism.FunctionSpace import construct_function_space
from optimism.Mechanics import MechanicsFunctions
from optimism.Mechanics import create_multi_block_mechanics_functions
from optimism.Mesh import Mesh
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
    U = self.create_field(Uu, p)
    state = p[1]
    element_stiffnesses = self.mech_funcs.compute_element_stiffnesses(U, state)
    fspace = self.mech_funcs.fspace
    return SparseMatrixAssembler.\
        assemble_sparse_stiffness_matrix(element_stiffnesses, fspace.mesh.conns, self.dof)

  def create_field(self, Uu, p):
    return self.dof.create_field(Uu, self.get_ubcs(p))

  def create_unknowns(self):
    Uu = self.dof.get_unknown_values(np.zeros(self.mesh.coords.shape))
    return Uu

  def energy_function(self, Uu, p):
    U = self.create_field(Uu, p)
    state = p[1]
    return self.mech_funcs.compute_strain_energy(U, state)

  # hardcoded for now TODO
  def get_ubcs(self, p):
    yLoc = p[0]
    V = np.zeros(self.mesh.coords.shape)
    if self.disp_nset is not None:
      index = (self.mesh.nodeSets[self.disp_nset], 1)
      V = V.at[index].set(yLoc)
    return self.dof.get_bc_values(V)


