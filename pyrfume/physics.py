from math import exp
from math import log as ln

import quantities as pq
from sympy import Symbol
from sympy.solvers import solve


def mackay(vp):
    """Mackay, D., & van Wesenbeeck, I. (2014).
    Correlation of Chemical Evaporation Rate with Vapor Pressure.
    Environmental Science & Technology, 48(17), 10259–10263.
    doi:10.1021/es5029074
    Note typo in intercept parameter in Figure 1.
    """
    # Units of Pascals
    vp = vp.rescale(pq.Pa)
    # Strip units for logarithm
    vp /= vp.units
    # # Evaporation rate
    if vp > 0:
        er = exp(1.0243 * ln(vp) - 15.08)
    else:
        er = 0
    # Attach units
    er *= pq.mol / (pq.m ** 2 * pq.s)
    return er


def bernoulli(v=None, p=None, rho=None, g=9.8 * pq.m / (pq.s) ** 2, z=0, k=None):
    g = float(g.simplified)
    if v is None:
        v = Symbol("v", real=True, positive=True)
    else:
        v = v.rescale(pq.m / pq.s)
        v = float(v.simplified)
    if p is None:
        p = Symbol("p", real=True, positive=True)
    else:
        p = p.rescale(pq.Pa)
        p = float(p.simplified)
    if rho is None:
        rho = Symbol("rho", real=True, positive=True)
    else:
        rho = rho.rescale(pq.kg / (pq.m ** 3))
        rho = float(rho.simplified)
    if k is None:
        k = Symbol("k", real=True, positive=True)
    else:
        k = k.rescale((pq.m / pq.s) ** 2)
        k = float(k.simplified)
    result = solve((v ** 2) / 2 + g * z + p / rho - k, [v, p, rho], dict=True)
    return result


def venturi(rho=None, p1=None, p2=None, v1=None, v2=None):
    assert rho is not None
    rho = rho.rescale(pq.kg / (pq.m ** 3))
    rho = float(rho.simplified)

    if v1 is None:
        v1 = Symbol("v1", real=True, positive=True)
    else:
        v1 = v1.rescale(pq.m / pq.s)
        v1 = float(v1.simplified)
    if v2 is None:
        v2 = Symbol("v2", real=True, positive=True)
    else:
        v2 = v2.rescale(pq.m / pq.s)
        v2 = float(v2.simplified)
    if p1 is None:
        p1 = Symbol("p1", real=True, positive=True)
    else:
        p1 = p1.rescale(pq.Pa)
        p1 = float(p1.simplified)
    if p2 is None:
        p2 = Symbol("p2", real=True, positive=True)
    else:
        p2 = p2.rescale(pq.Pa)
        p2 = float(p2.simplified)

    result = solve(rho * (v2 ** 2 - v1 ** 2) / 2 - p1 + p2, [v1, v2, p1, p2], dict=True)
    return result


if __name__ == "__main__":
    # print(bernoulli(None, 10*pq.psi, 1.225*pq.kg/(pq.m**3)))
    print(
        venturi(rho=1.225 * pq.kg / (pq.m ** 3), p1=10 * pq.psi, p2=1 * pq.psi, v1=10 * pq.m / pq.s)
    )
