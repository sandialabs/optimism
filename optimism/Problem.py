from .Domain import Domain
from optimism import EquationSolver as EqSolver
from optimism import Objective
from optimism import VTKWriter
import equinox as eqx
import jax.numpy as np


class Problem(eqx.Module):
  domain: Domain
  objective: Objective.Objective
  eq_settings: any # Get this type right

  def __init__(self, domain, eq_settings):
    self.domain = domain
    Uu = domain.create_unknowns()
    p = Objective.Params(
      bc_data=np.zeros((np.sum(self.domain.dof.isBc),)),
      state_data=domain.mech_funcs.compute_initial_state(),
      time=0.0
    )
    precond_strategy = Objective.PrecondStrategy(domain.assemble_sparse)
    self.objective = Objective.Objective(domain.energy_function, Uu, p, precond_strategy)
    self.eq_settings = eq_settings
    self.plot_solution(Uu, p, 'output', 0)

  def solve_load_step(self, Uu, p, n, useWarmStart=True):
    Uu, p = self._solve_load_step(Uu, p, useWarmStart)
    self.plot_solution(Uu, p, 'output', n)
    return Uu, p

  # TODO eventually this is what we want to jit
  # since we won't be able to jit the plotting step
  # for problems where we want gradients, we'll also want
  # to attempt to jit around those calculations as well 
  # so we only jit once
  def _solve_load_step(self, Uu, p, useWarmStart):
    # p = Objective.param_index_update(p, 0, disp)
    Uu,_ = EqSolver.nonlinear_equation_solve(
      self.objective, Uu, p, self.eq_settings,
      useWarmStart=useWarmStart
    )
    return Uu, p

  # TODO could probably have a better name
  def setup(self):
    Uu = self.domain.create_unknowns()
    p = Objective.Params(
      bc_data=np.zeros((np.sum(self.domain.dof.isBc),)),
      state_data=self.domain.mech_funcs.compute_initial_state(),
      time=0.0
    )
    return Uu, p

  # TODO add more outputs with optional flags for certain ones
  # e.g. properties, property gradients
  # state variables, element quantities, etc.
  # Should probably make a post-processor class
  # that is abstract but has derivable ones for e.g. exodus, vtk, etc.
  def plot_solution(self, Uu, p, plotBaseName, stepNum):
    # dispField = self.domain.create_field(Uu, p)
    dispField = self.domain.update_field(Uu, p)
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

  def update_field(self, Uu, p):
    return self.domain.update_field(Uu, p)

  def update_time(self, p, dt):
    return self.domain.update_time(p, dt)
