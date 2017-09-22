import numpy as np

def logistic(z):
    
    return 1 / (1 + np.exp(-z))

def logReg_cost(theta, X, y):
    '''Compute the cost function. 
    Inputs:
    X is an m by (n+1) numpy array, where m = # of training examples
                                     n = # of features (excluding the "1" column)
    y is an m by 1 numpy array
    theta is a 1 by (n+1) numpy array'''
    
    # first transform the inputs to numpy matrices
    X, y, theta = np.matrix(X), np.matrix(y), np.matrix(theta)
    
    ln_h = np.log( logistic(X*theta.T) )
    ln_1h = np.log( 1 - logistic(X*theta.T) ) 
    J_theta = ( -1/len(y) )*( y.T*ln_h + (1 - y.T)*ln_1h )
    
    return J_theta[0,0]