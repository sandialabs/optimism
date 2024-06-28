from optimism import Objective
from optimism import SparseMatrixAssembler
from optimism import VTKWriter
from optimism import EquationSolver as EqSolver
from optimism.FunctionSpace import DofManager, EssentialBC, FunctionSpace
from optimism.FunctionSpace import construct_function_space
from optimism.Mechanics import MechanicsFunctions, create_mechanics_functions
from optimism.Mesh import Mesh
from optimism.QuadratureRule import QuadratureRule
from optimism.QuadratureRule import create_quadrature_rule_1D, create_quadrature_rule_on_triangle
from typing import List, Optional
import equinox as eqx
import jax.numpy as np


class Domain(eqx.Module):
  mesh: Mesh
  ebcs: List[EssentialBC]
  mech_funcs: MechanicsFunctions
  dof: DofManager

  def __init__(
    self, mesh, ebcs, mat_model, mode2D, 
    p_order: Optional[int] = 1, # TODO hook this up
    q_order: Optional[int] = 2
  ):
    self.mesh = mesh
    self.ebcs = ebcs
    q_rule_1d = create_quadrature_rule_1D(q_order)
    q_rule_2d = create_quadrature_rule_on_triangle(q_order)
    fspace = construct_function_space(mesh, q_rule_2d)
    self.mech_funcs = create_mechanics_functions(fspace, mode2D, mat_model)
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
    index = (self.mesh.nodeSets['nset_outer_top'], 1)
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
    f_name = f'output-{str(0).zfill(5)}'
    plot_solution(self.domain, self.domain.create_field(Uu, p), f_name, p)

  def solve_load_step(self, Uu, p, n, disp):
    Uu, p = self._solve_load_step(Uu, p, disp)
    f_name = f'output-{str(n).zfill(5)}'
    plot_solution(self.domain, self.domain.create_field(Uu, p), f_name, p)
    return Uu, p

  # TODO eventually this is what we want to jit
  def _solve_load_step(self, Uu, p, disp):
    p = Objective.param_index_update(p, 0, disp)
    Uu,_ = EqSolver.nonlinear_equation_solve(self.objective, Uu, p, self.eq_settings)
    return Uu, p


def plot_solution(domain, dispField, plotName, p):
  writer = VTKWriter.VTKWriter(domain.mesh, baseFileName=plotName)
  writer.add_nodal_field(name='displacement',
                          nodalData=dispField,
                          fieldType=VTKWriter.VTKFieldType.VECTORS)
  
  bcs = np.array(domain.dof.isBc, dtype=int)
  writer.add_nodal_field(name='bcs',
                          nodalData=bcs,
                          fieldType=VTKWriter.VTKFieldType.VECTORS,
                          dataType=VTKWriter.VTKDataType.INT)
  
  writer.write()