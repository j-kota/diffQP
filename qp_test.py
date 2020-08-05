import numpy
import numpy as np
from numpy import hstack, vstack
from quadprog import solve_qp
import matplotlib.pyplot as plt


# This quadprog wrapper function was written by Stephane Caron, credits below
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):

    # Make-shift convert to double precision
    if P is not None:
        P = P.astype(np.float64)
    if q is not None:
        q = q.astype(np.float64)
    if G is not None:
        G = G.astype(np.float64)
    if h is not None:
        h = h.astype(np.float64)
    if A is not None:
        A = A.astype(np.float64)
    if b is not None:
        b = b.astype(np.float64)


    if initvals is not None:
        print("quadprog: note that warm-start values ignored by wrapper")
    qp_G = P
    qp_a = -q
    if A is not None:
        if G is None:
            qp_C = -A.T
            qp_b = -b
        else:
            qp_C = -vstack([A, G]).T
            qp_b = -hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T if G is not None else None
        qp_b = -h if h is not None else None
        meq = 0

    return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


# This script creates a soft sorting operator on the values 1,2,3.
# The order-3 permutahedron specified by these values is defined by the inequalities
#
# x+y+z = 6
# 1 <= x,y,z <= 3
#
# The projection of rho = [3,2,1] onto the permutahedron is modeled by q and P below
# (linear and quadratic parts, respectively)

if __name__ == "__main__":

    P = numpy.identity(3)
    P = .5 * (P + P.T)
    P = 1.0*P                                 #this is due to definition of P_Q
    eps = 0.00000001
    q = -1.0*numpy.array( [3.0, 2.0, 1.0] )

    G = numpy.array( [[ 1.0,  0.0,  0.0],
                      [ 0.0,  1.0,  0.0],
                      [ 0.0,  0.0,  1.0],
                      [-1.0,  0.0,  0.0],
                      [ 0.0, -1.0,  0.0],
                      [ 0.0,  0.0, -1.0]] )

    h = numpy.array( [3.0, 3.0, 3.0, -1.0, -1.0, -1.0] ).T

    # The equality constraints functioned correctly in quadprog
    # only when placed redundantly as such:
    A = numpy.array(  [[ 1.0,  1.0,  1.0],
                       [ 1.0,  1.0,  1.0]])

    b = numpy.array(  [6.0, 6.0]  ).T

    theta = quadprog_solve_qp(P, eps*q, G, h, A, b, None)

    print("theta[0] = ")
    print( theta[0] )
    print("theta[1] = ")
    print( theta[1] )

    sol_list = [ ( eps,quadprog_solve_qp(P, eps*q, G, h, A, b, None) ) for eps in numpy.linspace(0.0,2.0,100) ]
    x  = [  eps for (eps,q) in sol_list ]
    y1 = [ q[0] for (eps,q) in sol_list ]
    y2 = [ q[1] for (eps,q) in sol_list ]
    y3 = [ q[2] for (eps,q) in sol_list ]

    plotting = True
    if plotting:
        #plt.xlim(0,1000)
        #plt.ylim(0,22000)
        plt.plot(x, y1, 'r')
        plt.plot(x, y2, 'b')
        plt.plot(x, y3, 'y')
        plt.show()




# Below are the instructions for using quadprog_solve_qp() by Stephane Caron:
# The variable names used above as chosen to be consistent with those defined below:
"""
Solve a Quadratic Program defined as:
    minimize
        (1/2) * x.T * P * x + q.T * x
    subject to
        G * x <= h
        A * x == b
using quadprog <https://pypi.python.org/pypi/quadprog/>.
Parameters
----------
P : numpy.array
    Symmetric quadratic-cost matrix.
q : numpy.array
    Quadratic-cost vector.
G : numpy.array
    Linear inequality constraint matrix.
h : numpy.array
    Linear inequality constraint vector.
A : numpy.array, optional
    Linear equality constraint matrix.
b : numpy.array, optional
    Linear equality constraint vector.
initvals : numpy.array, optional
    Warm-start guess vector (not used).
Returns
-------
x : numpy.array
    Solution to the QP, if found, otherwise ``None``.
Note
----
The quadprog solver only considers the lower entries of `P`, therefore it
will use a wrong cost function if a non-symmetric matrix is provided.
"""







# Below is a copyright / disclaimer message from Stephane Caron, from whom
# the above quadprog_solve_qp() wrapper routine was borrowed:


# Copyright (C) 2016-2020 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.
