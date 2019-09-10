from __future__ import division
from builtins import map
from builtins import range
import numpy as np
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
from gpkit import Model, Variable, ConstraintSet, GPCOLORS, GPBLU
from gpkit.small_scripts import mag
from robust.robust_gp_tools import RobustGPTools


def plot_feasibilities(x, y, m, rm=None, iterate=False, skipfailures=False, numberofsweeps=150):
    """ Plots feasibility space over two variables given a model.

    Arguments
    ---------
    x : GPkit Variable (not a vector variable)
        plotted on the x axis
    y : GPkit Variable (not a vector variable)
        plotted on the y axis
    m : GPkit Model
    rm : GPkit Model (defaults to None)
        robust model, if it exists
    iterate : bool (defaults to False)
        whether or not to use iterable sweep
    skipfailures : bool (defaults to False)
        whether or not to skip errors during sweeps
    numberofsweeps : int (defaults to 150)
        number of points used to approximate the design space. If iterate = True, then this is the starting number of points.

    Raises
    ------
    ValueError if either x or y is a vector variable
    
    """
    interesting_vars = [x, y]
    rmtype = None
    if rm:
        rmtype = rm.type_of_uncertainty_set

    # posynomials = m.as_posyslt1()
    # old = []
    # while set(old) != set(interesting_vars):
    #     old = interesting_vars
    #     for p in posynomials:
    #         if set([var.key.name for var in interesting_vars]) & set([var.key.name for var in p.varkeys.keys()]):
    #             interesting_vars = list(set(interesting_vars) | set([m[var.key.name] for var in p.varkeys.keys() if var.key.pr is not None]))

    class FeasCircle(Model):
        """SKIP VERIFICATION"""

        def setup(self, m, sol, angles=None, rob=False):
            r = 4
            additional_constraints = []
            slacks = []
            thetas = []
            for count in range((len(interesting_vars) - 1)):
                th = Variable("\\theta_%s" % count, angles, "-") if angles is not None else Variable("\\theta_%s" % count, np.linspace(0, 2 * np.pi, numberofsweeps), "-")
                thetas += [th]
            for i_set in range(len(interesting_vars)):
                if rob:
                    eta_min_x, eta_max_x = RobustGPTools.generate_etas(interesting_vars[i_set])
                else:
                    eta_min_x, eta_max_x = 0, 0
                xo = mag(m.solution(interesting_vars[i_set]))
                x_center = np.log(xo)

                def f(c, index=i_set, x_val=x_center):
                    product = 1
                    for j in range(index):
                        product *= np.cos(c[thetas[j]])
                    if index != len(interesting_vars) - 1:
                        product *= np.sin(c[thetas[index]])
                    return np.exp(x_val) * np.exp(r * product)
                
                def g(c, index=i_set, x_val=x_center, x_nom=xo, eta=eta_max_x):
                    product = 1
                    for j in range(index):
                        product *= np.cos(c[thetas[j]])
                    if index != len(interesting_vars) - 1:
                        product *= np.sin(c[thetas[index]])
                    if rmtype == 'box':
                        return np.exp(max(r*np.abs(product) - (np.log(x_nom) + eta - x_val), 0))
                    return np.exp(np.abs((np.log(x_nom) + eta - x_val - r)*product))
                
                x_i = Variable('x_%s' % i_set, f, interesting_vars[i_set].unitstr())
                s_i = Variable("s_%s" % i_set)
                slacks += [s_i]

                uncertaintyset = Variable('uncertaintyset_%s' % i_set, g)
                var = RobustGPTools.variables_bynameandmodels(m, **interesting_vars[i_set].key.descr)

                if len(var) > 1:
                    raise ValueError("vector uncertain variables are not supported yet")
                else:
                    var = var[0]

                additional_constraints += [s_i >= 1, s_i <= uncertaintyset*1.000001, var / s_i <= x_i, x_i <= var * s_i]

            cost_ref = Variable('cost_ref', 1, m.cost.unitstr(), "reference cost")
            self.cost = sum([sl ** 2 for sl in slacks]) * m.cost / cost_ref
            feas_slack = ConstraintSet(additional_constraints)
            return [m, feas_slack], {k: v for k, v in list(sol["freevariables"].items())
                if k in m.varkeys and k.key.fix is True}
    
    def slope(a, b):
        if a[0] == b[0]:
            return 10e10 # arbitrarily large number
        return (b[1]-a[1])/(b[0]-a[0])

    def distance(a, b):
        return ((b[1]-a[1])**2+(b[0]-a[0])**2)**.5

    def angle(m1, m2):
        costheta = max((min(((-1-m1*m2)/((1+m1**2)*(1+m2**2))**.5, 1)),-1))
        return np.arccos(costheta)

    def iterate_angles(rob=False, spacing_start=150, slope_error=np.pi/4, dist_error=0.05):
        #TODO testing
        #TODO bug fix-slow? potentially modify slope error or dist error
        sol = rm.get_robust_model().solution if rob else m.solution
        spacing = np.linspace(0, 2 * np.pi, spacing_start+1)
        bounds = [(0, 2 * np.pi)]
        solved_angles = {}
        first = True
        while bounds:
            angles = spacing if first else [(bound[0]+bound[1])/2 for bound in bounds]
            fc = FeasCircle(m, sol, angles=angles, rob=rob)
            for interesting_var in interesting_vars:
                del fc.substitutions[interesting_var]
            feas = fc.solve(skipsweepfailures=skipfailures)

            p, q = list(map(mag, list(map(feas, [x, y]))))
            if not isinstance(p, np.ndarray):
                p = [p]
            if not isinstance(q, np.ndarray):
                q = [q]

            size = len(angles)
            for i in range(size):
                if first:
                    ccslope = slope([p[i], q[i]], [p[(i+1)%size], q[(i+1)%size]])
                    cslope = slope([p[(i-1)%size], q[(i-1)%size]], [p[i], q[i]])
                else:
                    ccslope = slope([p[i], q[i]], [solved_angles[bounds[i][1]][0], solved_angles[bounds[i][1]][1]])
                    cslope = slope([solved_angles[bounds[i][0]][0], solved_angles[bounds[i][0]][1]], [p[i], q[i]])
                solved_angles[angles[i]] = (p[i], q[i], ccslope, cslope)
            new_bounds = []
            for i in range(size):
                cur = angles[i]
                if first:
                    high = angles[(i+1)%size]
                    high_dist = distance(solved_angles[high], solved_angles[cur]) > dist_error
                    if np.pi-angle(solved_angles[cur][2], solved_angles[cur][3]) > slope_error and high_dist:
                        new_bounds += [(cur, high)]
                else:
                    low = bounds[i][0]
                    high = bounds[i][1]
                    low_dist = distance(solved_angles[low], solved_angles[cur]) > dist_error
                    high_dist = distance(solved_angles[high], solved_angles[cur]) > dist_error
                    if np.pi-angle(solved_angles[cur][2], solved_angles[cur][3]) > slope_error:
                        if low_dist:
                            new_bounds += [(low, cur)]
                        if high_dist:
                            new_bounds += [(cur, high)]
                    else:
                        if np.pi-angle(solved_angles[low][2], solved_angles[low][3]) > slope_error and low_dist:
                            new_bounds += [(low, cur)]
                        if np.pi-angle(solved_angles[high][2], solved_angles[high][3]) > slope_error and high_dist:
                            new_bounds += [(cur, high)]
            bounds = new_bounds
            if first:
                first = False

        angles = sorted(solved_angles)
        a = [solved_angles[angle][0] for angle in angles]
        b = [solved_angles[angle][1] for angle in angles]
        return a, b

    # plot original feasibility set
    # plot boundary of uncertainty set
    orig_a, orig_b, a, b = [None] * 4
    if iterate:
        if rm:
            a, b = iterate_angles(rob=True)
        orig_a, orig_b = iterate_angles()
    else:
        if rm:
            fc = FeasCircle(m, rm.get_robust_model().solution, rob=True)
            for interesting_var in interesting_vars:
                del fc.substitutions[interesting_var]
            rmfeas = fc.solve(skipsweepfailures=skipfailures)
            a, b = list(map(mag, list(map(rmfeas, [x, y]))))
        ofc = FeasCircle(m, m.solution)
        for interesting_var in interesting_vars:
            del ofc.substitutions[interesting_var]
        origfeas = ofc.solve(skipsweepfailures=skipfailures)
        orig_a, orig_b = list(map(mag, list(map(origfeas, [x, y]))))

    fig, axes = plt.subplots(2)

    def plot_uncertainty_set(ax):
        xo, yo = list(map(mag, list(map(m.solution, [x, y]))))
        ax.plot(xo, yo, "k.")
        if rm:
            eta_min_x, eta_max_x = RobustGPTools.generate_etas(x)
            eta_min_y, eta_max_y = RobustGPTools.generate_etas(y)
            x_center = np.log(xo)
            y_center = np.log(yo)
            ax.plot(np.exp(x_center), np.exp(y_center), "kx")
            if rmtype == "elliptical":
                th = np.linspace(0, 2 * np.pi, 50)
                ax.plot(np.exp(x_center) * np.exp(np.cos(th)) ** (np.log(xo) + eta_max_x - x_center),
                        np.exp(y_center) * np.exp(np.sin(th)) ** (np.log(yo) + eta_max_y - y_center), "k",
                        linewidth=1)
            elif rmtype:
                p = Polygon(
                    np.array([[xo * np.exp(eta_min_x)] + [xo * np.exp(eta_max_x)] * 2 + [xo * np.exp(eta_min_x)],
                              [yo * np.exp(eta_min_y)] * 2 + [yo * np.exp(eta_max_y)] * 2]).T,
                    True, edgecolor="black", facecolor="none", linestyle="dashed")
                ax.add_patch(p)

    if rm:
        axes[0].loglog([a[0]], [b[0]], color=GPCOLORS[1], linewidth=0.2)
    else:
        axes[0].loglog([orig_a[0]], [orig_b[0]], "k-")

    perimeter = np.array([orig_a, orig_b]).T
    p = Polygon(perimeter, True, color=GPBLU, linewidth=0)
    axes[0].add_patch(p)
    p = Polygon(perimeter, True, color=GPBLU, linewidth=0)
    axes[1].add_patch(p)

    if rm:
        perimeter = np.array([a, b]).T
        p = Polygon(perimeter, True, color=GPCOLORS[1], alpha=0.5, linewidth=0)
        axes[0].add_patch(p)
        p = Polygon(perimeter, True, color=GPCOLORS[1], alpha=0.5, linewidth=0)
        axes[1].add_patch(p)

    plot_uncertainty_set(axes[0])
    axes[0].axis("equal")
    axes[0].set_ylabel(y)
    plot_uncertainty_set(axes[1])
    axes[1].set_xlabel(x)
    axes[1].set_ylabel(y)

    fig.suptitle("%s vs %s Feasibility Space" % (x, y))
    plt.show()
