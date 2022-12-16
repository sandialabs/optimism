from optimism.JaxConfig import *
from optimism import FunctionSpace
from optimism import Interpolants
from optimism.material import LinearElastic as MatModel
from optimism import Mesh
from optimism import Mechanics
from optimism.EquationSolver import newton_solve
from optimism import QuadratureRule
from optimism import Surface
from optimism import TractionBC
from optimism.test import MeshFixture


E = 1.0
nu = 0.3
props = {'elastic modulus': E,
         'poisson ratio': nu,
         'strain measure': 'linear'}


class AxisymmPatchTest(MeshFixture.MeshFixture):
    
    def setUp(self):
        self.Nx = 3
        self.Ny = 2
        self.R = 1.0
        self.H = 1.0
        xRange = [0., self.R]
        yRange = [0., self.H]
        
        self.targetDispGrad = np.array([[-nu, 0.0],
                                        [0.0, 1.0]]) 
        
        self.mesh, self.U = self.create_mesh_and_disp(self.Nx, self.Ny, xRange, yRange,
                                                      lambda x : self.targetDispGrad.dot(x))
        quadRule = QuadratureRule.create_quadrature_rule_on_triangle(degree=2)
        self.fs = FunctionSpace.construct_function_space(self.mesh, quadRule, mode2D='axisymmetric')

        materialModel = MatModel.create_material_model_functions(props)
        
        mcxFuncs = \
            Mechanics.create_mechanics_functions(self.fs,
                                                 "axisymmetric",
                                                 materialModel)
        self.compute_energy = jit(mcxFuncs.compute_strain_energy)
        self.internals = mcxFuncs.compute_initial_state()


    def test_axisymmetric_gradient(self):
        elem0DispGrads = FunctionSpace.\
            compute_element_field_gradient(self.U, self.mesh.coords, self.fs.shapes[0],
                                           self.fs.shapeGrads[0], self.fs.vols[0], self.fs.mesh.conns[0],
                                           Mechanics.axisymmetric_element_gradient_transformation)

        # targetDispGrad is uniaxial stress state
        # additional strain should also be uniaxial Poisson contraction
        for dg in elem0DispGrads:
            self.assertNear(dg[2,2], -nu, 14)
        
        
    def test_dirichlet_patch_test(self):
        ebc = [FunctionSpace.EssentialBC(nodeSet='bottom', component=1),
               FunctionSpace.EssentialBC(nodeSet='top', component=1)]
        dofManager = FunctionSpace.DofManager(self.fs, dim=2, EssentialBCs=ebc)

        V = np.zeros(self.U.shape)
        V = V.at[self.mesh.nodeSets['top'],1].set(self.targetDispGrad[1,1])
        Ubc = dofManager.get_bc_values(V)
        
        @jit
        def objective(Uu):
            U = dofManager.create_field(Uu, Ubc)
            return self.compute_energy(U, self.internals)
        
        Uu = newton_solve(objective, dofManager.get_unknown_values(V))

        U = dofManager.create_field(Uu, Ubc)
            
        dispGrads = FunctionSpace.compute_field_gradient(self.fs, U)
        ne, nqpe = self.fs.vols.shape
        for dg in dispGrads.reshape(ne*nqpe,2,2):
            self.assertArrayNear(dg, self.targetDispGrad, 14)

        grad_func = jit(grad(objective))
        Uu = dofManager.get_unknown_values(self.U)
        self.assertNear(np.linalg.norm(grad_func(Uu)), 0.0, 14)

        energyComputed = objective(Uu)
        energyDensityExact = 0.5*E*self.targetDispGrad[1,1]**2
        energyExact = energyDensityExact*np.pi*self.R**2*self.H
        self.assertNear(energyComputed, energyExact, 14)

        
if __name__ == '__main__':
    MeshFixture.unittest.main()
