from matplotlib import pyplot as plt
import jax
from jax import numpy as np

from optimism import EquationSolver as EqSolver
from optimism import Mesh
from optimism import Mechanics
from optimism import FunctionSpace
from optimism.FunctionSpace import EssentialBC
from optimism.FunctionSpace import DofManager
from optimism import Objective
from optimism.ConstrainedObjective import ConstrainedObjective
from optimism.ConstrainedObjective import ConstrainedQuasiObjective
from optimism import QuadratureRule
from optimism import SparseMatrixAssembler
from optimism.Timer import Timer
from optimism import VTKWriter
from optimism import AlSolver

from optimism.test.MeshFixture import MeshFixture
from optimism.contact import Contact

from optimism import Surface


smoothingDistance = 1e-4
closest_distance_func = Contact.compute_closest_distance_to_each_side

settings = EqSolver.get_settings(use_incremental_objective=False,
                                 min_tr_size=1e-15,
                                 tol=1e-7)

alSettings = AlSolver.get_settings(max_gmres_iters=300,
                                   num_initial_low_order_iterations=5,
                                   use_second_order_update=True,
                                   penalty_scaling = 1.05,
                                   target_constraint_decrease_factor=0.5,
                                   tol=2e-7)

def write_output(mesh, dofManager, U, p, mechanicsFunctions, step, spheres=None, sphereRadii=None):
    plotName = get_output_name(step)
    writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)
    writer.add_nodal_field(name='displacement', nodalData=U, fieldType=VTKWriter.VTKFieldType.VECTORS)
    
    bcs = np.array(dofManager.isBc, dtype=int)
    writer.add_nodal_field(name='bcs', nodalData=bcs, fieldType=VTKWriter.VTKFieldType.VECTORS, dataType=VTKWriter.VTKDataType.INT)

    #internalVariables = mechanicsFunctions.compute_initial_state()
    #strainEnergyDensities, stresses = \
    #    mechanicsFunctions.\
    #    compute_output_energy_densities_and_stresses(U, internalVariables)
        
    #writer.add_cell_field(name='strain_energy_density',
    #                      cellData=strainEnergyDensities,
    #                      fieldType=VTKWriter.VTKFieldType.SCALARS)
    #writer.add_cell_field(name='stress',
    #                      cellData=stresses,
    #                      fieldType=VTKWriter.VTKFieldType.TENSORS)
    
    if spheres is not None:
        for s, sp in enumerate(spheres):
            writer.add_sphere(sp, sphereRadii[s])
    
    writer.write()


def write_debug_output(mesh, quadRule, edges2, U, p, subStep):
    plotName = get_iteration_output_name(subStep)
    writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)
    writer.add_nodal_field(name='displacement', nodalData=U,
                           fieldType=VTKWriter.VTKFieldType.VECTORS)

    quadPoints, dists = get_contact_points_and_field(edges2, mesh, U, p, quadRule, smoothingDistance)
    for e,edge in enumerate(quadPoints):
        for q,qpoint in enumerate(edge):
            writer.add_sphere(qpoint, dists[e][q])
    writer.write()
    
    
def get_output_name(N):
    return 'two-'+str(N).zfill(3)

    
def get_iteration_output_name(N):
    return 'substep-'+str(N).zfill(3)
    

def get_quadrature_points(edges, mesh, U, p, quadRule):
    def edge_coords(edge, mesh, coords, disp):
        index = Surface.get_field_index(edge, mesh.conns)
        endCoords = Surface.eval_field(coords, index) + Surface.eval_field(disp, index)
        qCoords = QuadratureRule.eval_at_iso_points(quadRule.xigauss, endCoords)    
        return qCoords
    
    return jax.vmap(edge_coords, (0,None,None,None))(edges, mesh, mesh.coords, U)


def get_contact_points_and_field(edges, mesh, U, p, quadRule, smoothingDistance):
    quadPoints = get_quadrature_points(edges, mesh, U, p, quadRule)
    interactionList = p[1][0]
    dists = closest_distance_func(mesh, U, quadRule, interactionList, edges, smoothingDistance)
    return quadPoints, dists


# This appears to be a fairly general quasi-static solver now.  Only for dense systems though.

def solve(Uu, energy_func, constraint_func, 
          update_params_func,
          write_output_func,
          write_debug_output_func,
          numSteps, settings, alSettings,
          initialMultiplier=4.0):

    step=0
    p = update_params_func(step, Uu, Objective.Params())

    c = constraint_func(Uu, p)
    kappa0 = initialMultiplier * np.ones_like(c)
    lam0 = 1e-4*np.abs(kappa0*c)
        
    objective = ConstrainedQuasiObjective(energy_func,
                                          constraint_func,
                                          Uu,
                                          p,
                                          lam0,
                                          kappa0)

    write_output_func(step, Uu, p, objective.lam)
    
    for step in range(1,numSteps+1):
        print('\n------------ LOAD STEP', step, '------------\n')
                
        count=0
        def iteration_plot(Uu, p):
            nonlocal count
            write_debug_output_func(count, Uu, p, objective.lam)
            count=count+1

        residuals=[]
        def subproblem_residual(Uu, obj):
            errorNorm = np.linalg.norm(obj.total_residual(Uu))
            residuals.append(errorNorm)
            print('error = ', errorNorm)
            with open('contact_residuals.'+str(count)+'.npz', 'wb') as file:
                np.savez(file,
                         data=np.array(residuals))
            
        p = update_params_func(step, Uu, p)
            
        Uu = AlSolver.augmented_lagrange_solve(objective, Uu, p, alSettings, settings, callback=iteration_plot, sub_problem_callback=subproblem_residual)

        write_output_func(step, Uu, p, objective.lam)

