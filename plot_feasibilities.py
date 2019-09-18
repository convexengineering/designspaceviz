from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from gpkit import Model, Variable, ConstraintSet, GPCOLORS, VectorVariable
from gpkit.small_scripts import mag

GPBLU, GPRED = GPCOLORS[:2]


def plot_feasibilities(axisvariables, m, rm=None, iterate=False, skipfailures=False,
                       numberofsweeps=150):
    """ Plots feasibility space over two variables given a model.

    Arguments
    ---------
    axisvariables : list of GPkit Variables (not a vector variable) 
        currently only 2 elements, first plotted on x, second plotted on y
    m : GPkit Model
    rm : GPkit Model (defaults to None)
        robust model, if it exists
    iterate : bool (defaults to False)
        whether or not to use iterable sweep
    skipfailures : bool (defaults to False)
        whether or not to skip errors during sweeps
    numberofsweeps : int (defaults to 150)
        number of points used to approximate the design space. If iterate=True,
        then this is the starting number of points.

    Raises
    ------
    ValueError if any of the axisvariables is a vector variable
    
    """
    axisvars = np.array(axisvariables)
    if rm:  # we have a robust model as well as a conventional one
        rmtype = rm.type_of_uncertainty_set
        from robust.robust_gp_tools import RobustGPTools

    class FeasCircle(Model):
        "Named model that will be swept to get feasibility boundary points."

        def setup(self, m, sol, angles=None):
            r = 4 # radius of the circle in logspace
            n_axes = len(axisvars)
            angles = angles or np.linspace(0, 2 * np.pi, numberofsweeps)
            thetas = np.array([Variable("\\theta_%s" % i, angles, "-") for i in range(n_axes - 1)])
            slacks = VectorVariable(n_axes, "s", "-", "slack variables")
            x = np.array([None]*n_axes, "object")  # filled in the loop below
            for i, var in enumerate(axisvars):
                if var.key.shape:
                    raise ValueError("Uncertain VectorVariables not supported")
                x_center = np.log(mag(m.solution(axisvars[i])))

                def f(c, index=i, x_val=x_center):
                    product = 1
                    for j in range(index):
                        product *= np.cos(c[thetas[j]])
                    if index != n_axes - 1: # TODO: should this be indented??
                        product *= np.sin(c[thetas[index]])
                    return np.exp(x_val) * np.exp(r * product)
                
                x[i] = Variable('x_%s' % i, f, axisvars[i].unitstr())
            
            constraints = [(slacks >= 1),
                          (axisvars/slacks <= x),
                          (axisvars*slacks >= x)]

            cost_ref = Variable('cost_ref', 1, m.cost.unitstr(), "reference cost")
            self.cost = (slacks**2).sum() * m.cost / cost_ref
            slack_constr = ConstraintSet(constraints)
            return ({"original model": m, "slack constraints": slack_constr},
                {k: v for k, v in list(sol["freevariables"].items())
                if k in m.varkeys and k.key.fix is True})

    def slope(a, b):
        if a[0] == b[0]:
            return 1e10  # arbitrarily large number. TODO: if np.inf, modify arithmetic in angle to account for possibility
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
            for axisvar in axisvars:
                del fc.substitutions[axisvar]
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

    # plot original feasibility set and boundary of uncertainty set
    orig_a, orig_b, a, b = [None] * 4
    if iterate:
        if rm:
            a, b = iterate_angles(rob=True)
        orig_a, orig_b = iterate_angles()
    else:
        if rm:
            fc = FeasCircle(m, rm.get_robust_model().solution)
            for axisvar in axisvars:
                del fc.substitutions[axisvar]
            rmfeas = fc.solve(skipsweepfailures=skipfailures)
            a, b = list(map(mag, list(map(rmfeas, axisvars))))
        ofc = FeasCircle(m, m.solution)
        for axisvar in axisvars:
            del ofc.substitutions[axisvar]
        origfeas = ofc.solve(skipsweepfailures=skipfailures)
        orig_a, orig_b = list(map(mag, list(map(origfeas, axisvars))))

    fig, axes = plt.subplots(2)

    def plot_uncertainty_set(ax):
        xo, yo = list(map(mag, list(map(m.solution, axisvars))))
        ax.plot(xo, yo, "k.")
        if rm:
            eta_min_x, eta_max_x = RobustGPTools.generate_etas(axisvars[0])
            eta_min_y, eta_max_y = RobustGPTools.generate_etas(axisvars[1])
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
        axes[0].loglog([a[0]], [b[0]], color=GPRED, linewidth=0.2)
    else:
        axes[0].loglog([orig_a[0]], [orig_b[0]], "k-")

    perimeter = np.array([orig_a, orig_b]).T
    p = Polygon(perimeter, True, color=GPBLU, linewidth=0)
    axes[0].add_patch(p)
    p = Polygon(perimeter, True, color=GPBLU, linewidth=0)
    axes[1].add_patch(p)

    if rm:
        perimeter = np.array([a, b]).T
        p = Polygon(perimeter, True, color=GPRED, alpha=0.5, linewidth=0)
        axes[0].add_patch(p)
        p = Polygon(perimeter, True, color=GPRED, alpha=0.5, linewidth=0)
        axes[1].add_patch(p)

    plot_uncertainty_set(axes[0])
    axes[0].axis("equal")
    axes[0].set_ylabel(axisvars[1])
    plot_uncertainty_set(axes[1])
    axes[1].set_xlabel(axisvars[0])
    axes[1].set_ylabel(axisvars[1])

    fig.suptitle("%s vs %s Feasibility Space" % (axisvars[0], axisvars[1]))
    return plt
