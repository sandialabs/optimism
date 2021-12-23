from optimism import Objective
from optimism.Objective import Params
from optimism.Objective import param_index_update
from optimism.test import TestFixture
from optimism.JaxConfig import *



class TestObjective(TestFixture.TestFixture):

    def test_param_change(self):
        params = Params(5)
        newParams = param_index_update(params, 0, 6)
        self.assertEqual( newParams, Params(6) )


    def test_param_change_first_out_of_two(self):
        params = Params(5, [41])
        newParams = param_index_update(params, 0, 6)
        self.assertEqual( newParams, Params(6, [41]) )


    def test_param_change_second_out_of_three(self):
        params = Params(5, [41], 'cat')
        newParams = param_index_update(params, 1, [42])
        self.assertEqual( newParams, Params(5, [42], 'cat'))

        
    def test_param_change_third_out_of_four(self):
        params = Params(5, [41], 'cat', {})
        newParams = param_index_update(params, 2, 'dog')
        self.assertEqual( newParams, Params(5, [41], 'dog', {}) )


    def test_param_change_four_out_of_four(self):
        params = Params(5, [41], 'cat', {})
        newParams = param_index_update(params, 3, [54,12])
        self.assertEqual( newParams, Params(5, [41], 'cat', [54,12]))

if __name__ == '__main__':
    TestFixture.unittest.main()
