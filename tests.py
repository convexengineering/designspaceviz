import unittest
import matplotlib as mpl
mpl.use('Agg')  # has to be called before matplotlib.pyplot is imported

from gpkit import settings
import gpkit.tests
from FeasibilityDemo import plot_feasibility_simple_Wing, C_Lmax, V_min


class TestFeasibilityDemo(unittest.TestCase):
    """test for the feasibility plots demo"""

    def test_elliptical(self):
        if settings["default_solver"] == "cvxopt":
            return
        plot_feasibility_simple_Wing('elliptical', [C_Lmax, V_min],
                                     'r', 1.25, 'r', 1.2)

    def test_box(self):
        if settings["default_solver"] == "cvxopt":
            return
        plot_feasibility_simple_Wing('box', [C_Lmax, V_min],
                                     'pr', 25, 'pr', 20)


TESTS = [TestFeasibilityDemo]


def run(xmloutput=False, verbosity=1):
    "run design space visualization tests"
    gpkit.tests.run(tests=TESTS, xmloutput=xmloutput, verbosity=verbosity)


if __name__ == "__main__":
    run()
