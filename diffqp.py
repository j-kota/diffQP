import torch
import torch.nn as nn
import numpy as np
from qp_test import quadprog_solve_qp
from sklearn.isotonic import IsotonicRegression
from scipy.linalg import block_diag


class diffqp(torch.autograd.Function):

    # Q,p,G,h,A,b are the data defining the QP problems to be solved
    # Each comes in a batch - dimension 0 of each tensor is the batch size
    # Definitions:
    # min (xT)Q(x) + px
    # st
    # Gx <= h
    # Ax = b
    @staticmethod
    def forward(ctx, Q_bat, p_bat, G_bat, h_bat, A_bat, b_bat):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        fwd_outputs  = [ qp_fwd(Q,p,G,h,A,b) for Q,p,G,h,A,b in zip(Q_bat, p_bat, G_bat, h_bat, A_bat, b_bat) ]

        fwd_sols    = [ a for (a,b) in fwd_outputs ]
        grad_info   = [ b for (a,b) in fwd_outputs ]

        grads       = [ qp_bkwd(b) for b in grad_info ]

        fwd_sols   = [ torch.Tensor( a ) for a in fwd_sols ]
        grads      = [ torch.Tensor( b ) for b in grads ]

        # the results are stacked to conform to batch processing format (first dim holds batch)
        ctx.save_for_backward(  torch.stack(grads)  )
        return torch.stack( fwd_sols )

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        jacobians, = ctx.saved_tensors
        N = len(grad_output)

        # multiply each gradient by the jacobian for the corresponding sample
        # then restack the results to preserve the batch gradients' format
        grad_input = torch.stack( [ torch.matmul( grad_output[i] , jacobians[i] ) for i in range(0,N) ] )

        return grad_input





# Non-torch function that carries out the forward algorithm
# Given a QP problem instance, returns the solution
#   along with necessary intermediate data to compute gradients
# Applies to one single training sample (one QP problem instance)
# Definitions:
# min (xT)Q(x) + px
# st
# Gx <= h
# Ax = b
# This function's inputs and outputs should be torch Tensors,
#   but intermediate calculations can take any form and use other libraries
def qp_fwd(Q,p,G,h,A,b):

    Q = Q.detach().numpy()
    p = p.detach().numpy()
    G = G.detach().numpy()
    h = h.detach().numpy()
    A = A.detach().numpy()
    b = b.detach().numpy()

    solution = quadprog_solve_qp( Q,p,G,h,A,b )
    grad_info = [0.0] # Not yet implemented
    return torch.Tensor(solution), torch.Tensor(grad_info)





# grad_info contains any data created in the forward pass
# which is required to calculate derivates for the backward pass
# This function's inputs and outputs should be torch Tensors,
#   but intermediate calculations can take any form and use other libraries
#
# Reference on the calculation of Jacobians:
# B.Amos 2017
# https://arxiv.org/abs/1703.00443
# Eqns (3)-(8)
def qp_bkwd(grad_info):
    jacobian = [[0.0]] # Not yet implemented
    return jacobian




# Initial main function - demonstrates a successful forward pass
#    on a batch of QP instances
if __name__ == "__main__":

    n = 5
    eps = 1e-3
    diffQP = diffqp()

    Q = eps*torch.eye(5)
    p = torch.ones(5)

    G = -torch.eye(5)
    h = -torch.Tensor( [1,2,3,4,5] )
    A = torch.zeros(5,5)
    b = torch.zeros(5)
    #A = torch.eye(5)
    #b = torch.Tensor( [1,2,3,4,5] )

    # Create batches, of size 1
    Q,p,G,h,A,b = ( torch.stack([Q]),
                    torch.stack([p]),
                    torch.stack([G]),
                    torch.stack([h]),
                    torch.stack([A]),
                    torch.stack([b])   )

    x = diffQP.apply( Q,p,G,h,A,b )

    print("Solving the QP:\n")
    print(" min (xT)Q(x) + px ")
    print(" st ")
    print(" Gx <= h ")
    print(" Ax = b ")
    print('\n')
    print("where \n")
    print("Q = ")
    print( Q )
    print("p = ")
    print( p )
    print("G = ")
    print( G )
    print("h = ")
    print( h )
    print("A = ")
    print( A )
    print("b = ")
    print( b )
    print('\n')
    print("solution:")
    print("x = ", x)
