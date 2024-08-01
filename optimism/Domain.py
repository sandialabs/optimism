from optimism import Objective
from optimism import SparseMatrixAssembler
from optimism import VTKWriter
from optimism import EquationSolver as EqSolver
from optimism.FunctionSpace import DofManager, EssentialBC, FunctionSpace
from optimism.FunctionSpace import construct_function_space
from optimism.Mechanics import MechanicsFunctions, create_mechanics_functions
from optimism.Mechanics import create_multi_block_mechanics_functions
from optimism.Mesh import Mesh
from optimism.QuadratureRule import QuadratureRule
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


class Problem(eqx.Module):
  domain: Domain
  objective: Objective.Objective
  eq_settings: any # Get this type right

  def __init__(self, domain, eq_settings):
    self.domain = domain
    Uu = domain.create_unknowns()
    p = Objective.Params(0.0, domain.mech_funcs.compute_initial_state())
    precond_strategy = Objective.PrecondStrategy(domain.assemble_sparse)
    self.objective = Objective.Objective(domain.energy_function, Uu, p, precond_strategy)
    self.eq_settings = eq_settings
    self.plot_solution(Uu, p, 'output', 0)

  def solve_load_step(self, Uu, p, n, disp):
    Uu, p = self._solve_load_step(Uu, p, disp)
    self.plot_solution(Uu, p, 'output', n)
    return Uu, p

  # TODO eventually this is what we want to jit
  # since we won't be able to jit the plotting step
  # for problems where we want gradients, we'll also want
  # to attempt to jit around those calculations as well 
  # so we only jit once
  def _solve_load_step(self, Uu, p, disp):
    p = Objective.param_index_update(p, 0, disp)
    Uu,_ = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)
    return Uu, p

  # TODO could probably have a better name
  def setup(self):
    Uu = self.domain.create_unknowns()
    p = Objective.Params(0.0, self.domain.mech_funcs.compute_initial_state())
    return Uu, p

  # TODO add more outputs with optional flags for certain ones
  # e.g. properties, property gradients
  # state variables, element quantities, etc.
  # Should probably make a post-processor class
  # that is abstract but has derivable ones for e.g. exodus, vtk, etc.
  def plot_solution(self, Uu, p, plotBaseName, stepNum):
    dispField = self.domain.create_field(Uu, p)
    plotName = f'{plotBaseName}-{str(stepNum).zfill(6)}'
    writer = VTKWriter.VTKWriter(self.domain.mesh, baseFileName=plotName)
    writer.add_nodal_field(
      name='displacement',
      nodalData=dispField,
      fieldType=VTKWriter.VTKFieldType.VECTORS
    )
    bcs = np.array(self.domain.dof.isBc, dtype=int)
    writer.add_nodal_field(
      name='bcs',
      nodalData=bcs,
      fieldType=VTKWriter.VTKFieldType.VECTORS,
      dataType=VTKWriter.VTKDataType.INT
    )
    writer.write()
