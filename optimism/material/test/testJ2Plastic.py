import unittest
from matplotlib import pyplot as plt
from jax.scipy.linalg import expm

from optimism.JaxConfig import *
from optimism import EquationSolver as EqSolver
from optimism import FunctionSpace
from optimism.material import J2Plastic as J2
from optimism import Mechanics
from optimism import Mesh
from optimism import Objective
from optimism import QuadratureRule
from optimism import TensorMath
from optimism.test.TestFixture import TestFixture
from optimism.test.MeshFixture import MeshFixture


plotting=False


def make_disp_grad_from_strain(strain):
    return expm(strain) - np.identity(3)
        

class GradOfPlasticityModelFixture(TestFixture):
    def setUp(self):
    
        E = 100.0
        poisson = 0.321
        Y0 = 0.3*E
        H = 1.0e-2*E

        self.props = {'elastic modulus': E,
                      'poisson ratio': poisson,
                      'yield strength': Y0,
                      'hardening model': 'linear',
                      'hardening modulus': H}

        materialModel = J2.create_material_model_functions(self.props)

        self.energy_density = materialModel.compute_energy_density
        self.stress_func = grad(self.energy_density, 0)
        self.compute_state_new = materialModel.compute_state_new
        self.tangents_func = hessian(self.energy_density)
        self.compute_initial_state = materialModel.compute_initial_state
        
    def test_zero_point(self):
        dispGrad = np.zeros((3,3))
        state = self.compute_initial_state()

        energy = self.energy_density(dispGrad, state)
        self.assertNear(energy, 0.0, 12)

        stress = self.stress_func(dispGrad, state)
        self.assertArrayNear(stress, np.zeros((3,3)), 12)


    def test_elastic_energy(self):
        strainBelowYield = 0.5*self.props['yield strength']/self.props['elastic modulus']
        
        strain = strainBelowYield*np.diag(np.array([1.0, -self.props['poisson ratio'], -self.props['poisson ratio']]))
        dispGrad = make_disp_grad_from_strain(strain)
        
        state = self.compute_initial_state()

        energy = self.energy_density(dispGrad, state)
        WExact = 0.5*self.props['elastic modulus']*strainBelowYield**2
        self.assertNear(energy, WExact, 12)

        F = dispGrad + np.identity(3)
        kirchhoffStress = self.stress_func(dispGrad, state) @ F.T
        kirchhoffstressExact = ops.index_update(np.zeros((3,3)),
                                       ops.index[0,0],
                                       self.props['elastic modulus']*strainBelowYield)
        self.assertArrayNear(kirchhoffStress, kirchhoffstressExact, 12)

        
    def test_elastic_strain_path(self):
        strain = np.zeros((3,3))
        strain_inc = 1e-6
        stateOld = self.compute_initial_state()

        strainHistory = []
        stressHistory = []
        eqpsHistory = []
        energyHistory = []
        for i in range(10):
            dispGrad = make_disp_grad_from_strain(strain)
            F = dispGrad + np.identity(3)
            energy = self.energy_density(dispGrad, stateOld)
            strainHistory.append(strain[0,0])
            stateNew = self.compute_state_new(dispGrad, stateOld)
            stressNew = self.stress_func(dispGrad, stateNew) @ F.T
            stressHistory.append(stressNew[0,0])
            eqpsHistory.append(stateNew[J2.EQPS])
            energyHistory.append(energy)

            stateOld = stateNew
            strain = ops.index_add(dispGrad, ops.index[0,0], strain_inc)

        E = self.props['elastic modulus']
        nu = self.props['poisson ratio']
        stressNewExp = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0*nu)) * strainHistory[-1]
        self.assertNear(stressHistory[-1], stressNewExp, 12)
        self.assertArrayNear(eqpsHistory, np.zeros(len(eqpsHistory)), 12)

        
    def test_plastic_strain_path(self):
        strain = np.zeros((3,3))
        strain_inc = 0.2
        stateOld = self.compute_initial_state()
        
        strainHistory = []
        stressHistory = []
        tangentsHistory = []
        eqpsHistory = []
        energyHistory = []
        for i in range(2):
            strain = ops.index_add(strain, ops.index[0,0], strain_inc)
            dispGrad = make_disp_grad_from_strain(strain)
            F = dispGrad + np.identity(3)
            energy  = self.energy_density(dispGrad, stateOld)
            tangentsNew = self.tangents_func(dispGrad, stateOld)
            stateNew = self.compute_state_new(dispGrad, stateOld)
            stressNew = self.stress_func(dispGrad, stateOld) @ F.T
            strainHistory.append(strain[0,0])
            stressHistory.append(stressNew[0,0])
            tangentsHistory.append(tangentsNew[0,0,0,0])
            eqpsHistory.append(stateNew[J2.EQPS])
            energyHistory.append(energy)

            stateOld = stateNew

        if plotting:
            plt.figure()
            plt.plot(strainHistory, energyHistory, marker='o', fillstyle='none')
            plt.xlabel('strain')
            plt.ylabel('potential density')
            
            plt.figure()
            plt.plot(strainHistory, stressHistory, marker='o', fillstyle='none')
            plt.xlabel('strain')
            plt.ylabel('stress')
            
            plt.figure()
            plt.plot(strainHistory, eqpsHistory, marker='o')
            plt.xlabel('strain')
            plt.ylabel('Eq Pl Strain')
            
            plt.figure()
            plt.plot(strainHistory, tangentsHistory, marker='o')
            plt.xlabel('strain')
            plt.ylabel('Tangent modulus')
            
            plt.show()

        E = self.props['elastic modulus']
        nu = self.props['poisson ratio']
        mu = 0.5*E/(1.0 + nu)
        lam = E*nu/(1.0 + nu)/(1.0 - 2.0*nu)
        Y0 = self.props['yield strength']
        H = self.props['hardening modulus']
        strainEnd = strainHistory[-1]
        # for solutions, refer to jax-fem/papers/plane_strain_unit_test.pdf
        eqpsExp = [max(0.0, (2.0 * i * mu - Y0) / (3.0*mu + H) ) for i in strainHistory]
        stressNewExp = (2.0 * mu * Y0 + \
            2.0 * strainEnd * pow(mu,2) + \
            strainEnd * H * lam + \
            2.0 * strainEnd * H * mu + \
            3.0 * strainEnd * lam * mu) \
            / (3.0 * mu + H)

        self.assertNear( stressHistory[-1], stressNewExp, 11 )
        self.assertArrayNear(eqpsHistory, eqpsExp, 12)
       

class PlasticityOnMesh(MeshFixture):

    def test_plasticity_with_mesh(self):

        dispGrad0 =  np.array([[0.4, -0.2],
                               [-0.04, 0.68]])
        mesh, U = self.create_mesh_and_disp(4,4,[0.,1.],[0.,1.],
                                            lambda x: dispGrad0@x)
        
        E = 100.0
        poisson = 0.321
        H = 1e-2 * E
        Y0 = 0.3 * E

        props = {'elastic modulus': E,
                 'poisson ratio': poisson,
                 'yield strength': Y0,
                 'hardening model': 'linear',
                 'hardening modulus': H}

        materialModel = J2.create_material_model_functions(props)

        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(1)
        fs = FunctionSpace.construct_function_space(mesh, quadRule)
        
        mechFuncs = Mechanics.create_mechanics_functions(fs,
                                                         "plane strain",
                                                         materialModel)
                
        EBCs = []
        EBCs.append(Mesh.EssentialBC(nodeSet='all_boundary', field=0))
        EBCs.append(Mesh.EssentialBC(nodeSet='all_boundary', field=1))
        dofManager = Mesh.DofManager(mesh, U.shape, EBCs)

        quadratureRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=1)
        fs = FunctionSpace.construct_function_space(mesh, quadratureRule)
        
        Ubc = dofManager.get_bc_values(U)

        nElems = Mesh.num_elements(mesh)
        nQpsPerElem = QuadratureRule.len(quadratureRule)
        internalVariables = mechFuncs.compute_initial_state()
        p = Objective.Params(None,
                             internalVariables)
        
        
        def compute_energy(Uu, p):
            U = dofManager.create_field(Uu, Ubc)
            internalVariables = p[1]
            return mechFuncs.compute_strain_energy(U, internalVariables)

        UuGuess = 0.0*dofManager.get_unknown_values(U)
        
        objective = Objective.Objective(compute_energy, UuGuess, p)
        
        Uu = EqSolver.nonlinear_equation_solve(objective, UuGuess, p, EqSolver.get_settings(), useWarmStart=False)

        U = dofManager.create_field(Uu, Ubc)

        dispGrads = FunctionSpace.compute_field_gradient(fs, U)
        for dg in dispGrads.reshape(nElems*nQpsPerElem, 2, 2):
            self.assertArrayNear(dg, dispGrad0, 11)

        internalVariablesNew = mechFuncs.compute_updated_internal_variables(U, p[1])
        
        # check to make sure plastic strain evolved
        # if this fails, make the applied displacement grad bigger
        eqpsField = internalVariablesNew[:,:,J2.EQPS]
        self.assertTrue(eqpsField[0] > 1e-8)

        
if __name__ == '__main__':
    unittest.main()
