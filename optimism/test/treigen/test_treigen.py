from optimism.JaxConfig import *
from optimism.treigen import treigen
from testutils import *
from jax.numpy.linalg import norm

import numpy as onp



def test_tr_spd_and_inside():
    A = np.array( [np.array([1.0,2.0,0.3]),
                   np.array([2.0,4.5,0.0]),
                   np.array([0.3,0.0,5.0])] )
    b = np.array( onp.random.random(3) )

    Delta = 100
    s = treigen.solve(A, b, Delta)

    assert_array_approx(A@s, -b)
    assert( b@s < 0 )
    assert( norm(s) < Delta )


def test_tr_spd_and_outside():
    A = np.array( [np.array([1.0,2.0,0.3]),
                   np.array([2.0,4.5,0.0]),
                   np.array([0.3,0.0,5.0])] )
    b = np.array( onp.random.random(3) )
    Delta = 1e-4
    s = treigen.solve(A, b, Delta)
    assert( b@s < 0 )
    assert( norm(s) == approx(Delta) )


def test_tr_not_spd_and_outside():
    A = np.array( [np.array([1.0,2.0,0.3]),
                   np.array([2.0,4.5,0.0]),
                   np.array([0.3,0.0,-5.0])] )
    b = np.array( onp.random.random(3) )
    Delta = 1e-4
    s = treigen.solve(A, b, Delta)
    assert( b@s < 0 )
    assert( norm(s) == approx(Delta) )


def test_tr_the_not_so_hard_case():
    A = np.array( [np.array([1.0, 2.0, 0.0]),
                   np.array([2.0, 4.5, 0.0]),
                   np.array([0.0, 0.0,-1.1])] )
    b = np.array([0.,0.0,1.])

    Delta = 1e4
    s = treigen.solve(A, b, Delta)
    assert( b@s < 0 )
    assert( norm(s) == approx(Delta) )


def test_tr_the_easy_hard_case():
    A = np.array( [np.array([1.0, 2.0, 0.0]),
                   np.array([2.0, 4.5, 0.0]),
                   np.array([0.0, 0.0,-1.1])] )
    b = np.array([2.0,1.0,0.0])

    Delta = 1e-2
    s = treigen.solve(A, b, Delta)

    assert( b@s < 0 )
    assert( norm(s) == approx(Delta) )


def test_tr_the_hard_case():
    A = np.array( [np.array([1.0, 2.0, 0.0]),
                   np.array([2.0, 4.5, 0.0]),
                   np.array([0.0, 0.0,-1.1])] )
    b = np.array([2.0,1.0,0.0])

    Delta = 1e3
    s = treigen.solve(A, b, Delta)

    assert( norm(s) == approx(Delta) )
    
