from pytest import approx

def assert_array_approx(a,b):
    assert(len(a)==len(b))
    for i in range(len(a)):
        assert a[i] == approx(b[i])
