import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from qp_test import quadprog_solve_qp
from sklearn.isotonic import IsotonicRegression
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt



class permuter(nn.Module):

    def __init__(self, input_size, eps):
        super(permuter,self).__init__()

        self.eps = eps

        self.maxval = input_size
        self.srank  = soft_rank
        self.relu = nn.ReLU()


        # This matrix is square, made of a stacked repeated row [1,2,...,N]
        # Where n is the length of an input
        # It's used to calculate a permutation matrix in forward()
        self.t = torch.stack( [torch.Tensor( range(0,self.maxval) ) + 1
                                            for _ in range(0,self.maxval) ]  )

    def forward(self,x):

        #ranks = self.srank(x, regularization_strength=0.001)
        ranks = self.srank(x, regularization_strength=self.eps)
        print("ranks[0] = ")
        print( ranks[0] )

        return self.relu( 1.0 - torch.abs( ranks[:,None] - self.t.T ) ).permute(0,2,1)

        # This demos how to use the output:
        # out = torch.bmm( x[:,None],M ).squeeze()


# TODO: Take the portion of the Jacobian only applying to z (not lambda)
# make sure Jacobian is transposed correctly

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
        #grad_info   = [ b for (a,b) in fwd_outputs ]
        #grads       = [ qp_bkwd(b) for b in grad_info ]

        #fwd_sols   = [ torch.Tensor( a ) for a in fwd_sols ]
        #grads      = [ torch.Tensor( b ) for b in grads ]


        grads       = [ b for (a,b) in fwd_outputs ]

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
        #grad_input = torch.stack( [ torch.matmul( grad_output[i] , jacobians[i] ) for i in range(0,N) ] )

        grad_input = ( torch.stack( [ torch.matmul( grad_output[i] , torch.zeros(jacobians[0].shape) ) for i in range(0,N) ] ) ,
                       torch.stack( [ torch.matmul( grad_output[i] , torch.zeros(jacobians[0].shape) ) for i in range(0,N) ] ) ,
                       torch.stack( [ torch.matmul( grad_output[i] , torch.zeros(jacobians[0].shape) ) for i in range(0,N) ] ) ,
                       torch.stack( [ torch.matmul( grad_output[i] , jacobians[i] ) for i in range(0,N) ] ) ,
                       torch.stack( [ torch.matmul( grad_output[i] , torch.zeros(jacobians[0].shape) ) for i in range(0,N) ] ) ,
                       torch.stack( [ torch.matmul( grad_output[i] , torch.zeros(jacobians[0].shape) ) for i in range(0,N) ] )   )


        return grad_input



# Non-torch function that carries out the forward algorithm
#   and calculates jacobians for the backward propagation
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
#
# Reference on the calculation of Jacobians:
# B.Amos 2017
# https://arxiv.org/abs/1703.00443
# Eqns (3)-(8)
def qp_fwd(Q,p,G,h,A,b):

    Q_m = matrix( (Q.T).tolist() )
    p_m = matrix( (p.T).tolist() )
    G_m = matrix( (G.T).tolist() )
    h_m = matrix( (h.T).tolist() )
    #A_m = matrix( (A.T).tolist() )
    #b_m = matrix( (b.T).tolist() )
    A_m = matrix( [] )
    b_m = matrix( [] )


    #solution = quadprog_solve_qp( Q,p,G,h,A,b )

    # Note - A is rejected when it doesn't have full rank
    #        empty A is also rejected
    #        Is there an alternative to leaving out A?
    qpsol  = solvers.qp( Q_m,p_m,G_m,h_m ) #,A,b )

    primal = torch.Tensor( np.array(qpsol['x']) ).flatten()
    dual   = torch.Tensor( np.array(qpsol['z']) ).flatten()



    # Form linear system to solve for the argmin gradients
    # Amos 2017 eqn (6)
    # form this to find only dz*/dh for now
    LHS = torch.cat(
            (torch.cat( (Q, G.T), 1 ),
             torch.cat( (torch.mm(torch.diag(dual),G),
                         torch.diag(torch.mv(G,primal)-h) ), 1) )
            ,0 )

    """
    LHS = torch.cat(
            (torch.cat( (Q, G.T, A.T), 1 ),
             torch.cat( (torch.mm(torch.diag(dual),G),
                         torch.diag(torch.mv(G,primal)-h),
                         torch.zeros(G.shape[0],A.shape[0])), 1),
             torch.cat( (A, torch.zeros(A.shape[0], G.shape[0]+A.shape[0])), 1) )
            ,0 )
    """

    # This RHS is specific to solving for dz*/dh
    h_RHS = torch.cat(  ( torch.zeros( len(dual),len(dual) ),                       #(torch.zeros( Q.shape[0], Q.shape[1]+G.shape[0]+A.shape[0] ),
                          torch.diag(dual) ),
                         0  )

    jacobi_z_h = torch.mm( torch.inverse(LHS), h_RHS )

    #get the component describing z
    jacobi_z_h = jacobi_z_h[:len(primal),:]



    grad_info = [0.0] # Not yet implemented
    # Need


    return primal, jacobi_z_h


class qpNet(nn.Module):
    def __init__(self, n):
        super(qpNet, self).__init__()

        self.fc1 = nn.Linear(n,n)
        self.bn1 = nn.BatchNorm1d(n)

        eps = 1e-3
        self.diffQP = diffqp()


        self.Q = eps*torch.eye(5)
        self.p = torch.ones(5)
        self.G = -torch.eye(5)
        #self.h = -torch.Tensor( [1,2,3,4,5] )
        self.A = torch.zeros(5,5)
        self.b = torch.zeros(5)

    def forward(self,x):

        l = self.fc1( x )
        #l = self.bn1( l )

        Q = torch.stack( [self.Q for _ in x] )
        p = torch.stack( [self.p for _ in x] )
        G = torch.stack( [self.G for _ in x] )
        A = torch.stack( [self.A for _ in x] )
        b = torch.stack( [self.b for _ in x] )

        """
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
        """

        out = self.diffQP.apply( Q, p, G, l , A, b )
        return out

def train():
    # Hyperparameters
    batch_size = 1
    num_epochs = 1000
    learning_rate = 1e-3
    # Data
    training_x = torch.Tensor( [[35,24,57,44,12]] )
    training_y = training_x        #torch.Tensor( [[0,24,59,103]] )
    n = len( training_x[0] )
    train_tensor = data.TensorDataset( training_x, training_y )
    train_loader = data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = False)
    # Model
    model = qpNet(n)
    criterion = nn.MSELoss()
    test_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(  model.parameters(), lr = learning_rate)

    losses = []
    for epoch in range(num_epochs):
            epoch_losses = []
            for i, (images, labels) in enumerate(train_loader):
                # Fwd
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Bkwd
                optimizer.zero_grad()
                temp = loss.backward()
                optimizer.step()

                print("epoch "+str(epoch)+", loss = "+str(loss.item()))

            losses.append(loss.item())

    print("final fc1 parameters:")
    print( model.fc1.weight.data )

    loss_list = ((np.array( losses ) / np.max(losses)) ).tolist()
    #percent_list = ((np.array( percent_list ) / np.max(percent_list)) ).tolist()
    plt.plot( range(1,len(loss_list)+1), loss_list, 'r' )
    #plt.plot( range(1,len(percent_list)+1), percent_list, 'b' )
    plt.show()


# demonstrates a successful forward pass
#    on a (singleton) batch of QP instances
def fwd_test():
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



if __name__ == "__main__":
    train()
