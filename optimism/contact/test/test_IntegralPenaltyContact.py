from optimism.contact.IntegralPenaltyContact import *

import unittest
import optimism.contact.SlidePlot as SlidePlot
from jax import numpy as np
from matplotlib import pyplot
from functools import partial

class TestEdgeIntersection(unittest.TestCase):

    def setUp(self):
        self.penalty_length = 0.1
        self.edge_smoothing = 0.2
        self.delta = 0.5
        self.xi = np.array([0.2, 0.9])
        return


    def utest_plot(self):
        g = np.linspace(1e-1, 1.2, 300)
        def p(g):
            return g + 1.0/g - 2
        pyplot.plot(g, p(g))
        pyplot.show()


    def test_integral_some_in_overlap(self):
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([-0.1, 0.1]), self.delta), np.inf,)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([0.1, -0.1]), self.delta), np.inf)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([-0.1, -0.1]), self.delta), np.inf)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([-0.2, 0.6]), self.delta), np.inf,)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([0.7, -0.3]), self.delta), np.inf)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([0.7, 0.0]), self.delta), np.inf)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([0.0, 0.7]), self.delta), np.inf)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([0.2, 0.0]), self.delta), np.inf)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([0.0, 0.3]), self.delta), np.inf)
        self.assertAlmostEqual(integrate_gap(self.xi, np.array([0.0, 0.0]), self.delta), np.inf)


    def test_integral_both_in_contact(self):
        g = np.array([0.1, 0.3])
        integral1 = integrate_gap(self.xi, g, self.delta)
        integral2 = integrate_gap(self.xi, g[::-1], self.delta)
        # g[::-1] flips the order of the values in the array
        self.assertAlmostEqual(integral1, integral2, 13)

        integral3 = integrate_gap_numeric(self.xi, g, self.delta)
        self.assertAlmostEqual(integral1, integral3, 7)

        integral4 = integrate_gap_numeric(self.xi, g[::-1], self.delta)
        self.assertAlmostEqual(integral2, integral4, 7)


    def test_integral_both_equal_in_contact(self):
        g = np.array([0.3, 0.3])
        integral1 = integrate_gap_numeric(self.xi, g, self.delta)
        integral2 = integrate_gap(self.xi, g, self.delta)
        self.assertAlmostEqual(integral1, integral2, 13)
        g = g.at[1].add(1e-8)
        integral3 = integrate_gap(self.xi, g, self.delta)
        self.assertAlmostEqual(integral1, integral3, 7)


    def test_integral_both_out_of_contact(self):
        g = np.array([0.6, 0.8])
        integral1 = integrate_gap(self.xi, g, self.delta)
        integral2 = integrate_gap(self.xi, g[::-1], self.delta)
        self.assertAlmostEqual(integral1, 0.0, 13)
        self.assertAlmostEqual(integral2, 0.0, 13)
        integral3 = integrate_gap_numeric(self.xi, g, self.delta)
        self.assertAlmostEqual(integral1, integral3, 7)


    def test_integral_both_equal_out_of_contact(self):
        g = np.array([0.8, 0.8])
        integral1 = integrate_gap(self.xi, g, self.delta)
        integral2 = integrate_gap(self.xi, g[::-1], self.delta)
        self.assertAlmostEqual(integral1, 0.0, 13)
        self.assertAlmostEqual(integral2, 0.0, 13)
        integral3 = integrate_gap_numeric(self.xi, g, self.delta)
        self.assertAlmostEqual(integral1, integral3, 7)


    def test_integral_in_out_of_contact(self):
        g = np.array([0.1, 0.7])
        integral1 = integrate_gap(self.xi, g, self.delta)
        integral2 = integrate_gap(self.xi, g[::-1], self.delta)
        self.assertAlmostEqual(integral1, integral2, 13)
        integral3 = integrate_gap_numeric(self.xi, g, self.delta)
        self.assertAlmostEqual(integral1, integral3, 7)


    def utest_debug(self):

        edgeA = np.array([[ 0.55,  0.05      ],
                          [ 0.50, -0.01 ]] )
        #edgeB = np.array([[0.1, 0. ],
        #                  [0.8, 1.4 ]])
        edgeB = np.array([[0.0, 0. ],
                          [1.0, 0.  ]])

        #edgeA = np.array([[ 0.53061224, 0.05],[0.59183673, -0.17346939]])
        #edgeB = np.array([[0.8, 0. ], [1.4, 0. ]])

        #normal_calc = compute_average_normal
        #normal_calc = compute_normal_from_b
        normal_calc = compute_normal_from_b

        energy1 = integrate_mortar_barrier(edgeA, edgeB, normal_calc, self.penalty_length)
        print("example energy = ", energy1)


    def test_plot_smoothness(self):
        
        self.penalty_length = 0.1
        self.edge_smoothing = 0.25

        normal_calc = compute_normal_from_b

        x0a = np.array([ 0.0, 0.05])
        #setup = 'flat', 'slant', 'perp', 'thinning'
        setup = 'thinning'
        if setup=='slant' or setup=='thinning':
          x0b = np.array([-1.0, 0.03]) # -0.025, 0.03, 0.12, 0.05
        elif setup=='flat':
          x0b = np.array([-1.0, 0.05])
        elif setup=='perp':
          x0b = np.array([-1.0, -0.2])

        edge0 = np.array([x0a, x0b])

        #slides to the right
        if (setup=='slant' or setup=='flat'):
            velocities = np.array([[0.2, 0.0], [0.2, 0.0]])
        elif setup=='perp':
            velocities = np.array([[0.2, 0.0], [0.6, 0.01]])
        elif setup=='thinning':
            velocities = np.array([[0.2, 0.0], [0.3, 0.0]])

        x1a = np.array([0.1, 0.0])
        x1b = np.array([0.8, 0.0])
        edge1 = np.array([x1a, x1b])
        edge2 = np.array([[0.8, 0.0], [1.4, 0.0]])
        edge3 = np.array([[1.4, 0.0], [2.4, 0.0]])

        N = 500
        S = np.linspace(0.0, 13.0, N)

        fig, axs = pyplot.subplots(1, 3, gridspec_kw={'width_ratios' : [2.,1.,1.]})

        def edge_0(step, edge0_mod):
            s = S[step]
            edge0_mod = edge0_mod.at[0,0].add(s*velocities[0,0])
            edge0_mod = edge0_mod.at[0,1].add(s*velocities[0,1])
            edge0_mod = edge0_mod.at[1,0].add(s*velocities[1,0])
            edge0_mod = edge0_mod.at[1,1].add(s*velocities[1,1])
            return edge0_mod

        def edge_1(step, edge_mod):
            return edge_mod

        def edge_2(step, edge_mod):
            return edge_mod
        
        def edge_3(step, edge_mod):
            return edge_mod

        def mesh0(step, edge0_mod=edge0):
            edge0_mod = edge_0(step, edge0_mod)
            return edge0_mod[:,0], edge0_mod[:,1]
        
        def mesh1(step, edge1_mod=edge1):
            edge1_mod = edge_1(step, edge1_mod)
            return edge1_mod[:,0], edge1_mod[:,1]

        def mesh2(step, edge2_mod=edge2):
            edge2_mod = edge_2(step, edge2_mod)
            return edge2_mod[:,0], edge2_mod[:,1]
        
        def mesh3(step, edge3_mod=edge3):
            edge3_mod = edge_3(step, edge3_mod)
            return edge3_mod[:,0], edge3_mod[:,1]

        def energy_pert(step, pert, e, n, d):
            es = np.array([edge_0(step, edge0), edge_1(step, edge1), edge_2(step, edge2), edge_3(step, edge3)])
            es = es.at[e,n,d].add(pert)
            energy1 = integrate_mortar_barrier(es[0], es[1], normal_calc, self.penalty_length, self.edge_smoothing)
            energy2 = integrate_mortar_barrier(es[0], es[2], normal_calc, self.penalty_length, self.edge_smoothing)
            energy3 = integrate_mortar_barrier(es[0], es[3], normal_calc, self.penalty_length, self.edge_smoothing)
            return energy1 + energy2 + energy3

        force = jax.grad(energy_pert, argnums=1)

        def energy(step):
            return energy_pert(step, 0.0, 0, 0, 0)

        times = np.arange(N)
        energies = jax.vmap(energy)(times)

        f000 = -jax.vmap(force, (0,None,None,None,None))(times, 0.0, 0, 0, 0)
        f001 = -jax.vmap(force, (0,None,None,None,None))(times, 0.0, 0, 0, 1)
        f010 = -jax.vmap(force, (0,None,None,None,None))(times, 0.0, 0, 1, 0)
        f011 = -jax.vmap(force, (0,None,None,None,None))(times, 0.0, 0, 1, 1)

        fs = np.array([[f000,f001],[f010,f011]])

        def arrows0(step, edge0_mod=edge0):
            edge0_mod = edge_0(step, edge0_mod)
            return edge0_mod[:,0], edge0_mod[:,1], [f000[step],f010[step]], [f001[step],f011[step]]

        def energy_vs_time(step):
            return S,energies
  
        def energy_at_time(step):
            s = S[step]
            return [s],[energies[step]]
        
        def force_vs_time(step, e, n, d):
            return S,fs[n,d]
        
        def force_at_time(step, e, n, d):
            return [S[step]],[fs[n,d][step]]

        p = SlidePlot.slide_plot(fig, axs, times)
        p.plot(0, mesh0, 'ko-')
        p.plot(0, mesh1, 'ko-')
        p.plot(0, mesh2, 'ko-')
        p.plot(0, mesh3, 'ko-')
        p.arrow(0, arrows0)

        axs[0].set_aspect('equal')
        axs[0].set_xlim([-1.0, 3.0])
        axs[0].set_ylim([-0.81, 0.81])
        axs[0].set_yticks(np.linspace(-0.8, 0.8, 17))

        p.plot(1, energy_vs_time, 'k')
        p.plot(1, energy_at_time, 'go')

        axs[1].legend(['energy'])

        p.plot(2, partial(force_vs_time,e=0,n=0,d=0), 'r--')
        p.plot(2, partial(force_at_time,e=0,n=0,d=0), 'ro')
        p.plot(2, partial(force_vs_time,e=0,n=0,d=1), 'r')
        p.plot(2, partial(force_at_time,e=0,n=0,d=1), 'ro')
        p.plot(2, partial(force_vs_time,e=0,n=1,d=0), 'g--')
        p.plot(2, partial(force_at_time,e=0,n=1,d=0), 'go')
        p.plot(2, partial(force_vs_time,e=0,n=1,d=1), 'g')
        p.plot(2, partial(force_at_time,e=0,n=1,d=1), 'go')

        axs[2].legend(['force_x r', '_nolegend_', 'force_y r', '_nolegend_', 'force_x l', '_nolegend_', 'force_y l', '_nolegend_'])
        # axs[2].set_ylim([-2, 1.2])

        p.show()

        #pyplot.clf()
        #x = np.linspace(-1.5,1.5,300)
        #pyplot.plot(x, spline_ramp(x))
        #pyplot.show()

unittest.main()
