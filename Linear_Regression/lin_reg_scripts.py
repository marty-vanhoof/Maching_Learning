def compute_cost(X, y, theta):
    '''Compute the cost function. 
    Inputs:
    X is an m by (n+1) numpy matrix, where m = # of training examples
                                     n = # of features (excluding the "1" column)
    y is an m by 1 numpy matrix
    theta is a 1 by n numpy matrix'''
    
    squared_residuals = np.power(X*theta.T - y, 2)
    return np.sum(squared_residuals) / (2 * len(X))

def gradient_descent(X, y, theta, alpha, num_iterations):
    '''Implement the gradient descent algorithm.
    Inputs:
    X is an m by (n+1) numpy matrix, where m = # of training examples
                                     n = # of features (excluding the "1" column)
    y is an m by 1 numpy matrix
    theta is a 1 by n numpy matrix
    alpha is the learning rate
    num_iterations is how many times we want to update the parameters theta'''
    
    m = len(y)
    # store the values of compute_cost() on each iteration
    cost_history = np.empty(num_iterations)
    
    for k in range(0, num_iterations):
        residuals = X*theta.T - y
        # update rule 
        theta = theta - (alpha / m) * (residuals.T)*X
        cost_history[k] = compute_cost(X, y, theta)
        
    return theta, cost_history