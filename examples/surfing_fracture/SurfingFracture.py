from matplotlib import pyplot as plt
from scipy.sparse import diags as sparse_diags
import numpy as onp

from optimism.JaxConfig import *

from optimism import BoundConstrainedObjective
from optimism import ConstrainedObjective
from optimism import EquationSolver as EqSolver
from optimism import BoundConstrainedSolver
from optimism import AlSolver
from optimism import FunctionSpace
from optimism import Mesh
from optimism.Mesh import DofManager
from optimism import Objective
from optimism import QuadratureRule
from optimism import ReadMesh
from optimism import SparseMatrixAssembler
from optimism import Surface
from optimism.Timer import Timer
from optimism import TensorMath
from optimism import VTKWriter
from optimism.phasefield import PhaseField
from optimism.phasefield import PhaseFieldLorentzPlastic as MatModel
import time

useCoarseMesh = False

E = 70.0e3
nu = 0.34
Y0 = 275.0
H = 1e-2*E
L = 0.01 if useCoarseMesh else 5e-4
L = 0.5*L
rpOverL = 3.0
Gc = 3.0*np.pi*Y0**2/(E/(1.0-nu**2))*L*rpOverL
ell = 5e-4
psiC = 3./16. * Gc/ell
void0 = 0.0 

props = {'elastic modulus': E,
         'poisson ratio': nu,
         'yield strength': Y0*1e5,
         'hardening model': 'linear',
         'hardening modulus': H,
         'critical energy release rate': Gc,
         'critical strain energy density': psiC,
         'regularization length': L,
         'void growth prefactor': 0.0,
         'void growth exponent': 1.0,
         'initial void fraction': 0.0}


mu = 0.5*E/(1.0+nu)
print('Gc = ', Gc)


def apply_mode_I_field_at_point(X, K_I):
    R = np.linalg.norm(X)
    theta = np.arctan2(X[1], X[0])    
    u0 = K_I/mu*np.sqrt(R/2.0/np.pi)*(1.0 - 2.0*nu + np.sin(0.5*theta)**2)*np.cos(0.5*theta)
    u1 = K_I/mu*np.sqrt(R/2.0/np.pi)*(2.0 - 2.0*nu - np.cos(0.5*theta)**2)*np.sin(0.5*theta)
    return np.array([u0, u1])


def apply_mode_I_Bc(boundaryNodeCoords, K_I):
    return vmap(apply_mode_I_field_at_point, (0, None))(boundaryNodeCoords, K_I)


crackDirection = np.array([1.0,0.0])
def compute_J_integral_on_edge(mesh, edge, W, stress, dispGrad):    
    SEM = W*np.eye(2) - np.dot(dispGrad.T, stress)
    t,n,jac = Surface.compute_edge_vectors(Surface.get_coords(mesh, edge))
    sign = n[0]*t[1] - n[1]*t[0]
    return sign*np.dot(crackDirection, SEM.dot(n))*jac


def J_integral(U, internals, mesh, fs, edges, bvpFuncs):

    dispGrads = FunctionSpace.compute_field_gradient(fs, U[:,:2])
    Ws, stresses = bvpFuncs.compute_output_energy_densities_and_fluxes(U, internals)

    dispGrads = FunctionSpace.project_quadrature_field_to_element_field(fs, dispGrads)
    stresses = FunctionSpace.project_quadrature_field_to_element_field(fs, stresses)
    Ws = FunctionSpace.project_quadrature_field_to_element_field(fs, Ws)
        
    computeJs = vmap(compute_J_integral_on_edge, (None, 0, 0, 0, 0))
    return np.sum(computeJs(mesh,
                            edges,
                            Ws[edges[:,0]],
                            stresses[edges[:,0]][:,:2,:2],
                            dispGrads[edges[:,0]][:,:2,:2]))

# solver settings

# for monolithic trust region

tolScale = 0.05

subProblemSettings = EqSolver.get_settings(max_trust_iters=500,
                                           tr_size=0.1,
                                           min_tr_size=1e-12,
                                           tol=5e-7*tolScale,
                                           cg_inexact_solve_ratio=5e-3,
                                           max_cg_iters=5,
                                           max_cumulative_cg_iters=20,
                                           use_preconditioned_inner_product_for_cg=False)

alSettings = AlSolver.get_settings(tol=2e-6*tolScale,
                                   target_constraint_decrease_factor=1.0,
                                   max_gmres_iters=50,
                                   use_second_order_update=False,
                                   num_initial_low_order_iterations=1)

# for alternating minimization

dispSettings = EqSolver.get_settings(max_trust_iters=100,
                                     tr_size=1.0,
                                     min_tr_size=1e-12,
                                     tol=5e-7*tolScale,
                                     max_cg_iters=50,
                                     use_preconditioned_inner_product_for_cg=False)

phaseSettings = EqSolver.get_settings(max_trust_iters=100,
                                      tr_size=0.005,
                                      min_tr_size=1e-12,
                                      tol=2e-7*tolScale,
                                      cg_inexact_solve_ratio=5e-3,
                                      max_cg_iters=5,
                                      max_cumulative_cg_iters=20,
                                      use_preconditioned_inner_product_for_cg=False)


phaseAlSettings = AlSolver.get_settings(tol=5e-7*tolScale,
                                        target_constraint_decrease_factor=1.0,
                                        max_gmres_iters=50,
                                        use_second_order_update=False,
                                        num_initial_low_order_iterations=10)

phaseStiffnessRelativeTolerance = 1.0/20.0


class SurfingProblem:

    def __init__(self):
        
        if useCoarseMesh:
            self.mesh = ReadMesh.read_json_mesh('surfingMeshCoarse.json')
        else:
            self.mesh = ReadMesh.read_json_mesh('surfingMesh.json')

        self.crackInc=5e-3
        self.KIc = np.sqrt(Gc*E/(1.-nu**2))
        self.loadSteps = 4
        
        # translation of K_I field origin
        EBCs = [Mesh.EssentialBC(nodeSet='external', field=0),
                Mesh.EssentialBC(nodeSet='external', field=1),
                Mesh.EssentialBC(nodeSet='ysymm', field=1),
                Mesh.EssentialBC(nodeSet='precrack', field=2),
                Mesh.EssentialBC(nodeSet='top', field=2)]

        nNodes = self.mesh.coords.shape[0]
        self.fieldShape = (nNodes, 3)
        self.dofManager = DofManager(self.mesh, self.fieldShape, EBCs)

        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule)
        self.nqp = QuadratureRule.len(quadRule)

        materialModel = MatModel.create_material_model_functions(props)
        self.bvpFunctions =  PhaseField.create_phasefield_functions(self.fs,
                                                                    "plane strain",
                                                                    materialModel)
        
        self.crackLengthHistory = []
        self.bcOriginHistory = []
        self.JHistory = []
                
        UIdsFull = self.dofManager.dofToUnknown.reshape(self.fieldShape)

        XIds = UIdsFull[self.dofManager.isUnknown[:,0],0]
        YIds = UIdsFull[self.dofManager.isUnknown[:,1],1]
        self.phaseIds = UIdsFull[self.dofManager.isUnknown[:,2],2]
        self.dispIds = np.sort(np.hstack((XIds, YIds)))
        
        
    def objective_function(self, Uu, p):
        U = self.create_field(Uu, p)
        internalVars = p[1]
        return self.bvpFunctions.compute_internal_energy(U, internalVars)
    
    
    def objective_function_phase(self, Uphase, p):
        Uu = ops.index_update(p[2], self.phaseIds, Uphase)
        U = self.create_field(Uu, p)
        internalVars = p[1]
        return self.bvpFunctions.compute_internal_energy(U, internalVars)

    
    def objective_function_disp(self, Udisp, p):
        Uu = ops.index_update(p[2], self.dispIds, Udisp)
        U = self.create_field(Uu, p)
        internalVars = p[1]
        return self.bvpFunctions.compute_internal_energy(U, internalVars)


    def assemble_objective_stiffness(self, Uu, p, useBlockDiagonal=False):
        U = self.create_field(Uu, p)
        internalVars = p[1]

        if useBlockDiagonal:
            elementKMatrices = self.bvpFunctions.\
                compute_block_diagonal_element_stiffnesses(U, internalVars)
        else:
            elementKMatrices = self.bvpFunctions.\
                compute_element_stiffnesses(U, internalVars)

        return SparseMatrixAssembler.assemble_sparse_stiffness_matrix(elementKMatrices,
                                                                      self.mesh.conns,
                                                                      self.dofManager)


    def assemble_phase_stiffness(self, Uphase, p):
        Uu = ops.index_update(p[2], self.phaseIds, Uphase)
        K = self.assemble_objective_stiffness(Uu, p) 
        return K[:,self.phaseIds][self.phaseIds,:]


    def assemble_disp_stiffness(self, Udisp, p):
        Uu = ops.index_update(p[2], self.dispIds, Udisp)
        K = self.assemble_objective_stiffness(Uu, p) 
        return K[:,self.dispIds][self.dispIds,:]
    
    
    def get_ubcs(self, p):
        nNodes = self.mesh.coords.shape[0]
        V = np.zeros((nNodes,3))

        index = ops.index[self.mesh.nodeSets['precrack'],2]
        V = ops.index_update(V, index, 1.0)

        KI, origin = p[0]
        Xb = self.mesh.coords[self.mesh.nodeSets['external'],:]
        Xb = ops.index_add(Xb, ops.index[:,0], -origin)
        modeIBcs = apply_mode_I_Bc(Xb, KI)
        index = ops.index[self.mesh.nodeSets['external'],:2]
        V = ops.index_update(V, index, modeIBcs)
                             
        return self.dofManager.get_bc_values(V)


    def create_field(self, Uu, p):
        return self.dofManager.create_field(Uu, self.get_ubcs(p))


    def plot_solution(self, U, p, lam, plotNameBase, step):
        plotName = plotNameBase + '-' + str(step).zfill(3)
        
        mesh = self.mesh
        dofManager = self.dofManager
        fs = self.fs
        bvpFuncs = self.bvpFunctions
        internalVars = p[1]

        writer = VTKWriter.VTKWriter(mesh, baseFileName=plotName)

        writer.add_nodal_field(name='displacement', nodalData=U[:,:2],
                               fieldType=VTKWriter.VTKFieldType.VECTORS)

        writer.add_nodal_field(name='phase', nodalData=U[:,2],
                               fieldType=VTKWriter.VTKFieldType.SCALARS)

        writer.add_nodal_field(name='bcs',
                               nodalData=np.array(dofManager.isBc, dtype=int),
                               fieldType=VTKWriter.VTKFieldType.VECTORS,
                               dataType=VTKWriter.VTKDataType.INT)

        writer.add_nodal_field(name='phase_multiplier', nodalData=lam,
                               fieldType=VTKWriter.VTKFieldType.SCALARS)

        eqpsField = internalVars[:,:,MatModel.STATE_EQPS]
        eqpsField = FunctionSpace.\
            project_quadrature_field_to_element_field(self.fs, eqpsField)
        writer.add_cell_field(name='eqps', cellData=eqpsField,
                              fieldType=VTKWriter.VTKFieldType.SCALARS)
        writer.write()

        a = 2.0 * bvpFuncs.compute_phase_potential_energy(U, internalVars) / Gc
        J = 2.0 * J_integral(U, internalVars, mesh, fs, mesh.sideSets['contour'], bvpFuncs)
        self.crackLengthHistory.append(float(a))

        #self.bcOriginHistory.append(float(p[0][1]))
        self.JHistory.append(float(J))
        with open(plotNameBase+'_r_curve.npz', 'wb') as file:
            np.savez(file,
                     crackLengthHistory=np.array(self.crackLengthHistory),
                     J=np.array(self.JHistory),
                     U=U)

    
    def run(self):
 
        KI = 0.0
        bcCrackTip = 0.0
        internalVars = self.bvpFunctions.compute_initial_state()
        
        # initial condition
 
        Uzero = np.zeros(self.fieldShape)
        Uu = self.dofManager.get_unknown_values(Uzero)

        initialCrackTipX=0.0
        pInitial = Objective.Params(np.array([KI, initialCrackTipX]),
                                    internalVars)
        
        p = Objective.Params(np.array([KI, bcCrackTip]), internalVars)
        
        precondStrategy = Objective.TwoTryPrecondStrategy(partial(self.assemble_objective_stiffness, useBlockDiagonal=False),
                                                          partial(self.assemble_objective_stiffness, useBlockDiagonal=True))
        
        objective = BoundConstrainedObjective.BoundConstrainedObjective(self.objective_function,
                                                                        Uu,
                                                                        p,
                                                                        self.phaseIds,
                                                                        constraintStiffnessScaling=phaseStiffnessRelativeTolerance,
                                                                        precondStrategy=precondStrategy)
        
        U = self.create_field(Uu, p)
        lamField = ops.index_update(np.zeros_like(U[:,0]),
                                    self.dofManager.isUnknown[:,2],
                                    objective.get_multipliers())
        
        numTrustRegionSolves = []
        runTimes = []

        for step in range(self.loadSteps):
            print("\n------------------------\n")
            print(" LOAD STEP: ", step)
            p = Objective.param_index_update(p, 0, np.array([KI, bcCrackTip]))
            #p = Objective.param_index_update(p, 1, internalVars)

            residuals = []
            
            subproblemSolveCount=0
            def subproblem_counter(UuBar, obj):
                nonlocal subproblemSolveCount
                subproblemSolveCount+=1
                if step==self.loadSteps-1:
                    errorNorm = np.linalg.norm(obj.total_residual(UuBar))
                    residuals.append(errorNorm)
                    print('al error = ', errorNorm)
                    with open('al_residuals.npz', 'wb') as file:
                        np.savez(file,
                                 data=np.array(residuals))
                
            t = time.time()
            
            Uu = BoundConstrainedSolver.bound_constrained_solve(objective,
                                                                Uu,
                                                                p,
                                                                alSettings,
                                                                subProblemSettings,                                                            
                                                                sub_problem_callback=subproblem_counter,
                                                                useWarmStart=False)
            dt = time.time()-t
            
            U = self.create_field(Uu, p)
            #internalVars = self.bvpFunctions.\
            #    compute_updated_internal_variables(U, p[1])
            lamField = ops.index_update(np.zeros_like(U[:,0]),
                                        self.dofManager.isUnknown[:,2],
                                        objective.get_multipliers())
            
            self.plot_solution(U,
                               p,
                               lamField,
                               "surfing",
                               step)
                        
            if step == 0:
                KI += self.KIc
            else:
                bcCrackTip += self.crackInc

            numTrustRegionSolves.append(subproblemSolveCount)
            runTimes.append(dt)
            with open('trust_iters.npz', 'wb') as file:
                np.savez(file,
                         numTrustRegionSolves=np.array(numTrustRegionSolves),
                         runTimes=np.array(runTimes))


                
    def run_alternating_min(self):

        KI = 0.0
        bcCrackTip = 0.0
        internalVars = self.bvpFunctions.compute_initial_state()

        # initial condition
        Uzero = np.zeros((self.mesh.coords.shape[0],3))
        Uu = self.dofManager.get_unknown_values(Uzero)
                
        UuPhase = Uu[self.phaseIds]
        UuDisp = Uu[self.dispIds]

        p = Objective.Params(np.array([KI, bcCrackTip]),
                             internalVars,
                             Uu)
        
        phasePrecondStrategy = Objective.PrecondStrategy(self.assemble_phase_stiffness)
        phaseObjective = BoundConstrainedObjective.BoundConstrainedObjective(self.objective_function_phase,
                                                                             UuPhase,
                                                                             p,
                                                                             np.arange(UuPhase.size),
                                                                             constraintStiffnessScaling=phaseStiffnessRelativeTolerance,
                                                                             precondStrategy=phasePrecondStrategy)

        dispPrecondStrategy = Objective.PrecondStrategy(self.assemble_disp_stiffness)
        dispObjective = Objective.ScaledObjective(self.objective_function_disp,
                                                  UuDisp,
                                                  p,
                                                  precondStrategy=dispPrecondStrategy)
        numAltMinSteps = []
        runTimes = []
        
        for step in range(self.loadSteps):
            print("\n------------------------\n")
            print(" LOAD STEP: ", step)
            print("\n------------------------\n")

            t = time.time()

            residuals = []

            pError = np.linalg.norm( phaseObjective.get_total_residual(UuPhase) )
            dispError = np.linalg.norm( dispObjective.get_residual(UuDisp) )
            tError = np.sqrt( pError**2 + dispError**2 )
            
            residuals.append( tError )
            
            # update params and state for phase
            maxIters = 1000
            for i in range(maxIters):
            
                phaseObjective.p = p
                
                print("Minimizing phase: objective = ", phaseObjective.get_value(UuPhase))

                UuPhase = BoundConstrainedSolver. \
                    bound_constrained_solve(phaseObjective, UuPhase, p,
                                            phaseAlSettings, phaseSettings,
                                            useWarmStart=False)
                Uu = ops.index_update(Uu, self.phaseIds, UuPhase)
                print("Minimized phase: objective = ", phaseObjective.get_value(UuPhase))
                
                p = Objective.param_index_update(p, 0, np.array([KI, bcCrackTip]))
                p = Objective.param_index_update(p, 1, internalVars)
                p = Objective.param_index_update(p, 2, Uu)
                
                dispObjective.p = p
                Udisp = Uu[self.dispIds]
                
                print("phase force error = ", np.linalg.norm(phaseObjective.get_residual(UuPhase)))
                print("disp force error = ", np.linalg.norm(dispObjective.get_residual(UuDisp)))
                print("------------------------------")
                
                print("Minimizing disp: objective = ", dispObjective.get_value(UuDisp))
                UuDisp = EqSolver.nonlinear_equation_solve(dispObjective,
                                                           UuDisp,
                                                           p,
                                                           dispSettings,
                                                           useWarmStart=False)
                
                Uu = ops.index_update(Uu, self.dispIds, UuDisp)
                print("Minimized disp: objective = ", dispObjective.get_value(UuDisp))

                p = Objective.param_index_update(p, 0, np.array([KI, bcCrackTip]))
                p = Objective.param_index_update(p, 1, internalVars)
                p = Objective.param_index_update(p, 2, Uu)
                phaseObjective.p = p
                
                phaseError = np.linalg.norm( phaseObjective.get_residual(UuPhase) )
                dispError = np.linalg.norm( dispObjective.get_residual(UuDisp) )

                totalError = np.sqrt( phaseError**2 + dispError**2 )

                if step==self.loadSteps-1:
                    pError = np.linalg.norm(phaseObjective.get_total_residual(UuPhase))
                    tError = np.sqrt( pError**2 + dispError**2 )
                    print('terror, total error = ', totalError, tError)
                    residuals.append(tError)
                    
                    with open('alt_residuals.npz', 'wb') as file:
                        np.savez(file,
                                 data=np.array(residuals))
                
                print('phase force error = ', phaseError)
                print('disp force error = ', dispError)
                print("------------------------------")
                if totalError < alSettings.tol:
                    break

            dt = time.time()-t
            
            U = self.create_field(Uu, p)
                
            internalVars = self.bvpFunctions.\
                compute_updated_internal_variables(U, p[1])
            
            lamField = ops.index_update(np.zeros_like(U[:,0]),
                                        self.dofManager.isUnknown[:,2],
                                        phaseObjective.get_multipliers())
            self.plot_solution(U,
                               p,
                               lamField,
                               "surfingAlt",
                               step)
            
            if step == 0:
                KI += self.KIc
            else:
                bcCrackTip += self.crackInc

            numAltMinSteps.append(i+1)
            runTimes.append(dt)
            with open('alternating_iters.npz', 'wb') as file:
                np.savez(file,
                         numAltMinSteps=np.array(numAltMinSteps),
                         runTimes=np.array(runTimes))
            
            
app = SurfingProblem()

#t0 = time.time()
app.run()
#trtime = time.time() - t0

#t0 = time.time()
app.run_alternating_min()
#alttime = time.time() - t0

