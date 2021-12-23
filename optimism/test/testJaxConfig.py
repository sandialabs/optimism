from optimism.JaxConfig import *
from optimism.test.TestFixture import *


class TestDebugIsOff(TestFixture):
    def test_debug_if_off(self):
        self.assertTrue(not jaxDebug)
