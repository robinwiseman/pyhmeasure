import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
from scipy.special import betainc
from typing import Dict
from inspect import ismethod


class CostRatioDensity:
    def __init__(self, c_density_obj=None):
        if c_density_obj is None:
            self.c_density_obj = beta(2,2) # default H-Measure choice
        else:
            if not (self.has_method(c_density_obj, 'pdf') and
                    self.has_method(c_density_obj, 'cdf')):
                raise NotImplementedError(f"""density object: {c_density_obj} is missing pdf or cdf""")
            self.c_density_obj = c_density_obj
        # For a Beta(a, b) density the integrals needed by the H-Measure are available in closed form
        # (as in the Rust implementation). Any other density is integrated numerically.
        self.beta_params = self._beta_params(self.c_density_obj)

    def __call__(self, cost):
        '''
        :param cost: is the cost ratio of the costs for incorrect classification of each class
        cost := (1+(c1/c0))^(-1), and so lies in [0,1]. Suitable distribution functions should be
        defined on the same domain.
        :return: the probability density of the cost
        '''
        return self.c_density_obj.pdf(cost)

    def uc(self, cost):
        return cost*self.c_density_obj.pdf(cost)

    def u1mc(self, cost):
        return (1-cost)*self.c_density_obj.pdf(cost)

    def cdf(self, cost):
        return self.c_density_obj.cdf(cost)

    def uc_integral(self, lo, hi):
        '''
        :return: the integral of cost*pdf(cost) over [lo, hi]. For a Beta(a, b) density, cost*pdf(cost) is
        proportional to the Beta(a+1, b) density, so the integral follows from the Beta(a+1, b) cdf.
        '''
        if self.beta_params is None:
            return quad(self.uc, lo, hi)[0]
        a, b = self.beta_params
        return a/(a+b)*(betainc(a+1, b, hi) - betainc(a+1, b, lo))

    def u1mc_integral(self, lo, hi):
        '''
        :return: the integral of (1-cost)*pdf(cost) over [lo, hi]. For a Beta(a, b) density, (1-cost)*pdf(cost) is
        proportional to the Beta(a, b+1) density, so the integral follows from the Beta(a, b+1) cdf.
        '''
        if self.beta_params is None:
            return quad(self.u1mc, lo, hi)[0]
        a, b = self.beta_params
        return b/(a+b)*(betainc(a, b+1, hi) - betainc(a, b+1, lo))

    @staticmethod
    def _beta_params(c_density_obj):
        # (a, b) if the density is a scipy Beta distribution on [0,1] created as beta(a, b), otherwise None
        dist = getattr(c_density_obj, 'dist', None)
        args = getattr(c_density_obj, 'args', ())
        kwds = getattr(c_density_obj, 'kwds', {})
        if getattr(dist, 'name', None) == 'beta' and len(args) == 2 and not kwds:
            return float(args[0]), float(args[1])
        return None

    @staticmethod
    def has_method(obj, method):
        return hasattr(obj, method) and ismethod(getattr(obj, method))


class HMeasure:
    def __init__(self, cost_distribution: CostRatioDensity, class0_prior: float = None,
                 class1_prior: float = None):
        self.cost_distribution = cost_distribution
        self.class0_prior = class0_prior
        self.class1_prior = class1_prior
        self.c_merged = None
        self.c_classes = None
        self.roc_curve = None
        self.convex_hull = None
        self.int_components = None
        self.H = None

    def h_measure(self, scores: Dict[str, np.array]):
        '''
        :param scores: a dict of class_0 and class_1 scores, each as numpy arrays
        :return:
        '''
        c0_scores = scores.get('class_0', None)
        c1_scores = scores.get('class_1', None)
        self.c0_num = len(c0_scores)
        self.c1_num = len(c1_scores)
        if (self.class0_prior is None) or (self.class1_prior is None):
            # Optionally pass in population priors (to handle case where model training is over/under sampled
            # for one class relative to the other.
            # If no priors are passed, then calculate priors from the score populations.
            self.class0_prior = self.c0_num/(self.c0_num+self.c1_num)
            self.class1_prior = self.c1_num/(self.c0_num+self.c1_num)
        c0_scores = np.sort(c0_scores)
        c1_scores = np.sort(c1_scores)
        if self.H is None:
            self.c_merged, self.c_classes = self._merge_scores(c0_scores, c1_scores)
            self.roc_curve = self._build_roc(self.c_merged, self.c_classes)
            self.convex_hull, self.int_components = self._build_chull(self.roc_curve)
            self.H = self._build_H(self.int_components)

        return self.roc_curve, self.convex_hull, self.H

    def _merge_scores(self, c0: np.array, c1: np.array):
        # plain python lists are faster than numpy arrays for element by element access
        c0 = c0.tolist()
        c1 = c1.tolist()
        c_merged_scores = [0.0]*(self.c0_num+self.c1_num)
        c_merged_classes = [0]*(self.c0_num+self.c1_num)
        c0_i = 0
        c1_i = 0
        cm_i = 0
        while (c0_i < self.c0_num) or (c1_i < self.c1_num):
            if c0_i < self.c0_num and c1_i < self.c1_num:
                if c0[c0_i] <= c1[c1_i]:
                    c_merged_scores[cm_i] = c0[c0_i]
                    c_merged_classes[cm_i] = 0
                    c0_i += 1
                    cm_i += 1
                elif c1[c1_i] <= c0[c0_i]:
                    c_merged_scores[cm_i] = c1[c1_i]
                    c_merged_classes[cm_i] = 1
                    c1_i += 1
                    cm_i += 1
            elif c0_i < self.c0_num and c1_i >= self.c1_num: # loop to fill in the remaining c0
                c_merged_scores[cm_i] = c0[c0_i]
                c_merged_classes[cm_i] = 0
                c0_i += 1
                cm_i += 1
            elif c1_i < self.c1_num and c0_i >= self.c0_num: # loop to fill in the remaining c1
                c_merged_scores[cm_i] = c1[c1_i]
                c_merged_classes[cm_i] = 1
                c1_i += 1
                cm_i += 1

        return c_merged_scores, c_merged_classes

    def _build_roc(self, c_scores, c_classes):
        # class1 on the 'x-axis', class0 on the 'y-axis' : [x,y]
        # Cumulative counts keep each coordinate an exactly rounded fraction, so that collinear points stay
        # collinear and the curve ends at exactly [1,1].
        rc_points = [[0.0, 0.0]]
        count_0 = 0
        count_1 = 0
        i = 0
        while i < len(c_scores):
            (duplicate_0, duplicate_1) = self.find_duplicate(c_scores,c_classes,i)
            count_0 += duplicate_0
            count_1 += duplicate_1
            rc_points.append([count_1/self.c1_num, count_0/self.c0_num])
            i += duplicate_0+duplicate_1

        return np.array(rc_points)

    def find_duplicate(self, c_scores, c_classes, i):
        dups_class0 = 0
        dups_class1 = 0
        for j in range(i,len(c_scores)):
            if c_scores[j] == c_scores[i]:
                if c_classes[j] == 0:
                    dups_class0 += 1
                else:
                    dups_class1 += 1
            elif c_scores[j] > c_scores[i]:
                break

        return (dups_class0, dups_class1)

    def _build_chull(self, roc_curve):
        # The roc curve points are already ordered along the curve, so the upper convex hull follows from a
        # single pass over them (a monotone chain): drop the last hull point while it lies on or below the line
        # from its predecessor to the incoming point.
        chull = [] # the coordinates of the convex hull in roc curve space
        for point in roc_curve.tolist():
            while len(chull) >= 2:
                o = chull[-2]
                q = chull[-1]
                cross = (q[0]-o[0])*(point[1]-o[1]) - (q[1]-o[1])*(point[0]-o[0])
                if cross >= 0.0:
                    chull.pop()
                else:
                    break
            chull.append(point)

        # Note: for points on the roc_curve, point[0] corresponds to class 1, point[1] to class 0.
        # Collect elements required for subsequent integration to generate the H-measure:
        # [c_{i+1},c_i,r1_i,r0_i] for each point i on the convex hull
        int_components = []
        cval_prior = 0.0
        for current_point, next_point in zip(chull[:-1], chull[1:]):
            # rounding on almost collinear edges can leave a c value marginally below its predecessor
            cval = max(self._cval(current_point, next_point), cval_prior)
            int_components.append([cval, cval_prior, current_point[0], current_point[1]])
            cval_prior = cval
        int_components.append([1.0, cval_prior, chull[-1][0], chull[-1][1]])

        return np.array(chull), np.array(int_components)

    def _cval(self, current_point, i_point):
        # Note: for points, point[0] corresponds to class 1, point[1] to class 0
        return self.class1_prior*(i_point[0]-current_point[0])/(self.class0_prior*(i_point[1]-current_point[1]) + self.class1_prior*(i_point[0]-current_point[0]))

    def _build_L(self, int_components):
        L = 0
        for i, int_vals in enumerate(int_components):
            L += self._int_L_step(int_vals)

        return L

    def _int_L_step(self, int_vals):
        r0i = int_vals[3]
        r1i = int_vals[2]
        ci = int_vals[1]
        cip1 = int_vals[0]
        int_uc = self.cost_distribution.uc_integral(ci, cip1)
        int_u1mc = self.cost_distribution.u1mc_integral(ci, cip1)

        L_step = self.class0_prior*(1-r0i)*int_uc + self.class1_prior*r1i*int_u1mc

        return L_step

    def _L_max(self):
        L_max = self.cost_distribution.uc_integral(0, self.class1_prior)*self.class0_prior
        L_max += self.cost_distribution.u1mc_integral(self.class1_prior, 1)*self.class1_prior
        return L_max

    def _build_H(self, int_vals):
        L = self._build_L(int_vals)
        L_max = self._L_max()
        return 1 - L/L_max
