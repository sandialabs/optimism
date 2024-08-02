from optimism import *
# from optimism.material import Neohookean as MatModel

mesh = read_exodus_mesh('./geometry.g')

ebcs = [
  EssentialBC(nodeSet='nset_outer_bottom', component=0),
  EssentialBC(nodeSet='nset_outer_bottom', component=1),
  EssentialBC(nodeSet='nset_outer_top', component=0),
  EssentialBC(nodeSet='nset_outer_top', component=1)
]

props = {
  'elastic modulus': 3. * 10.0 * (1. - 2. * 0.3),
  'poisson ratio'  : 0.3,
  'version'        : 'coupled'
}
mat_models = {
  'block_1': Neohookean.create_material_model_functions(props)
}

eq_settings = get_settings(
  use_incremental_objective=False,
  max_trust_iters=100,
  tr_size=0.25,
  min_tr_size=1e-15,
  tol=5e-8
)

domain = Domain(mesh, ebcs, mat_models, 'plane strain', disp_nset='nset_outer_top')
problem = Problem(domain, eq_settings)

# setup
Uu, p = problem.setup()

# iterate over load steps
disp = 0.0
n_steps = 20
ddisp = -0.2 / n_steps
for n in range(n_steps):
  print(f'Load step {n + 1}')
  disp += ddisp
  Uu, p = problem.solve_load_step(Uu, p, n + 1, disp)
