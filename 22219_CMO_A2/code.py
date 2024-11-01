import sys
sys.path.append('/Users/sacchitmac/Desktop/mac 2')
import os
import numpy as np
import matplotlib.pyplot as plt
from oracles_updated import f1, f2, f3

srn = 22219

# Conjugate Gradient Algorithm
def conjugate_gradient(A, b, x0=None, tol=1e-5, max_iter=1000):
    """
    Conjugate Gradient algorithm to solve Ax = b
    """
    n = b.shape[0]  # This should be the size of the vector b (which is 5)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0
    
    r = b - np.dot(A, x)  # Residual
    d = r.copy()          # Initial direction is the residual
    rs_old = np.dot(r.T, r)  # Dot product should yield a scalar automatically

    for k in range(max_iter):
        Ad = np.dot(A, d)  # Matrix-vector multiplication (A is 5x5, d is 5x1)
        alpha = rs_old / np.dot(d.T, Ad)  # Step size (ensure scalar)
        
        x = x + alpha * d  # Update the solution vector
        r = r - alpha * Ad  # Update the residual

        rs_new = np.dot(r.T, r)  # This should automatically be a scalar
        
        if np.sqrt(rs_new) < tol:  # Check for convergence
            print(f"Converged in {k+1} iterations")
            break

        beta = rs_new / rs_old  # Update for the direction
        d = r + beta * d  # Update the direction

        rs_old = rs_new
    
    return x, k+1


# Constant Gradient Descent
def constant_gradient_descent(alpha, oracle, max_iter=100, x_0=None):
    if x_0 is None:
        x_0 = np.zeros(5)
    f_values = np.zeros(max_iter+1)
    f_values[0] = oracle(x_0, srn, 0)
    iterations = 0
    while iterations < max_iter:
        x_0 = x_0 - alpha * oracle(x_0, srn, 1)
        iterations += 1
        f_values[iterations] = oracle(x_0, srn, 0)
    return x_0, f_values

# Newtons Method
def newton_method(oracle, max_iter=100, x_0=None):
    if x_0 is None:
        x_0 = np.zeros(5)
    f_values = np.zeros(max_iter + 1)
    f_values[0] = oracle(x_0, srn, 0)
    iterations = 0
    while iterations < max_iter:
        H_inv_grad = oracle(x_0, srn, 2)
        if np.any(np.isnan(H_inv_grad)) : # Check if the oracle returned NaN
            print("Oracle returned NaN at iteration:", iterations+1)
        if np.any(np.isinf(H_inv_grad)):  # Check if the oracle returned Inf
            print("Oracle returned Inf at iteration:", iterations+1)
        x_0 = x_0 - H_inv_grad
        iterations += 1
        f_values[iterations] = oracle(x_0, srn, 0)
    return x_0, f_values

# Rank-One Update
def rank_one_update(oracle, x_0=None, tol=1e-6, max_iter=100, c1=1e-4, c2=0.9, alpha_init=1.0, beta=0.5):
    if x_0 is None:
        x_0 = np.zeros(5)
    x = x_0
    f_values = np.zeros(max_iter + 1)  # Store function values
    f_values[0] = oracle(x, srn, 0)  # Initial function value
    n = len(x_0)
    B_k = np.eye(n)  # Initial Hessian approximation as identity matrix
    grad_k = oracle(x, srn, 1)  # order=1 returns the gradient
    i = 0  # Initialize iteration counter

    while i < max_iter:
        # Stopping condition if the norm of the gradient is smaller than tolerance
        if np.linalg.norm(grad_k)  < tol:
            print(f"Converged after {i} iterations")
            break
        
        # Compute the search direction
        p_k = -np.dot(B_k, grad_k)
        
        # Wolfe line search
        alpha = alpha_init
        f_x = oracle(x, srn, 0)
        grad_dot_pk = np.dot(grad_k.T, p_k)
        
        while True:
            f_x_alpha = oracle(x + alpha * p_k, srn, 0)
            grad_new = oracle(x + alpha * p_k, srn, 1)

            # Armijo (sufficient decrease) condition
            if f_x_alpha > f_x + c1 * alpha * grad_dot_pk:
                alpha *= beta  # Decrease alpha using a backtracking factor
                continue

            # Curvature (Wolfe) condition
            if np.dot(grad_new.T, p_k) < c2 * grad_dot_pk:
                alpha *= beta  # Decrease alpha if curvature condition is not met
                continue

            # If both conditions are satisfied, break
            break
        
        # Update x using the chosen alpha
        x_new = x + alpha * p_k
        # Compute s_k and y_k
        s_k = x_new - x
        y_k = grad_new - grad_k
        
        # Rank-one update for B_k
        s_y_diff = y_k - np.dot(B_k, s_k)
        B_k += np.outer(s_y_diff, s_y_diff) / np.dot(s_y_diff, s_k)
        
        # Move to next iteration
        x = x_new
        grad_k = grad_new
        i += 1
        f_values[i] = oracle(x, srn, 0)

    return x, f_values[:i+1]
def quasi_newton_scalar(oracle, x_0=None, tol=1e-8, max_iter=100, c1=1e-4, c2=0.9, alpha_init=1.0, beta=0.5):
    if x_0 is None:
        x_0 = np.zeros(5)  # Default initial point if not provided
    
    x = x_0
    f_values = np.zeros(max_iter + 1)  # Store function values for analysis
    f_values[0] = oracle(x, srn, 0)  # Initial function value
    grad_x = oracle(x, srn, 1)  # order=1 returns the gradient
    B_inv_scalar = 1.0  # Scalar approximation of the inverse Hessian (initially 1)

    i = 0
    while i < max_iter:
        # Stopping condition: If gradient norm is below the tolerance
        if np.linalg.norm(grad_x) < tol:
            print(f"Converged after {i} iterations")
            break
        
        # Compute the search direction
        p_k = -B_inv_scalar * grad_x
        
        # Armijo-Wolfe line search to determine step size alpha
        alpha = alpha_init
        f_x = oracle(x, srn, 0)
        grad_dot_pk = np.dot(grad_x.T, p_k)
        
        while True:
            f_x_alpha = oracle(x + alpha * p_k, srn, 0)
            grad_new = oracle(x + alpha * p_k, srn, 1)

            # Armijo (sufficient decrease) condition
            if f_x_alpha > f_x + c1 * alpha * grad_dot_pk:
                alpha *= beta  # Backtrack alpha if condition not met
                continue

            # Wolfe (curvature) condition
            if np.dot(grad_new.T, p_k) < c2 * grad_dot_pk:
                alpha *= beta  # Backtracking if curvature condition not met
                continue

            # If both conditions are satisfied, break out of line search
            break
        
        # Update x
        x_new = x + alpha * p_k
        
        # Compute s_k and y_k
        s_k = x_new - x
        grad_new = oracle(x_new, srn, 1)
        y_k = grad_new - grad_x
        
        # Update scalar B_inv using the secant equation
        if np.dot(s_k, y_k) != 0:
            B_inv_scalar = np.dot(s_k, y_k) / np.dot(y_k, y_k)
        
        # Update values for next iteration
        x = x_new
        grad_x = grad_new
        i += 1
        f_values[i] = oracle(x, srn, 0)

    return x, f_values[:i+1]



# Question 1.2
A,b = f1(srn,True) # psd matrix A for subq 2
x0 = np.zeros(b.shape)
x, num_iter = conjugate_gradient(A, b, x0)
x = x.reshape(x.shape[0],)
print(f'Optimal Solution = {[float(round(i, 4)) for i in x]}')
print(f'Number of Iterations = {num_iter}')



# Question 1.4
A,b = f1(srn,False)
Q = np.dot(A.T,A)
print(Q)
print('shape of Q:',Q.shape)
b_new = np.dot(A.T,b)
b_new = b_new.reshape(b_new.shape[0],)

x,num_iter = conjugate_gradient(Q,b_new)  
x = x.reshape(x.shape[0],)
print(f'Optimal Solution = {[float(round(i, 4)) for i in x]}')
print(f'Number of Iterations = {num_iter}')


#Question 2.1
alpha = [0.5, 0.1, 0.01, 0.001]  # Step sizes

# Set up the subplots grid (2 rows x 2 columns for 4 step sizes)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Loop through alpha values and plot results on subplots
for idx, i in enumerate(alpha):
    # Run gradient descent for each step size
    x_0, f_values = constant_gradient_descent(i, f2)
    # Select the corresponding subplot
    ax = axs[idx // 2, idx % 2]
    # Plot the convergence of function values over iterations
    ax.plot(np.arange(len(f_values)), f_values)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Function Value')
    ax.set_title(f'Convergence for Step Size = {i}')   
    # Print Results for each alpha value
    print(f'Value of x  after 100 iterations for Step Size {i} = {[float(round(j,4)) for j in x_0]}')
    print(f'function value after 100 iterations for Step Size {i} = {f_values[-1]}')
    print(f'Gradient Norm at Optimal Solution = {np.linalg.norm(f2(x_0, srn, 1))}') # Check if the norm of the gradient is close to 0
    print(f'Gradient at Optimal Solution = {f2(x_0, srn, 1)}') # Check if the gradient is close to 0Vector

# Adjust layout and display the subplots
plt.tight_layout()
# plt.savefig('2.1_Gradient_Descent.png')
plt.show()


#Question 2.2
x_0, f_values = newton_method(f2)
print(f'Optimal Solution after 100 iterations = {x_0}')
print(f'Function Value at Optimal Solution = {f_values[-1]}')
plt.plot(np.arange(len(f_values)), f_values)
plt.ylabel('Function Value')
plt.xlabel('Iterations')
plt.title('2.2 Convergence of Newton Method')
# plt.savefig('2.2_Newton_Method.png')
plt.show()


#Question 2.3
# Define 5 different initial values (starting points)
initial_values = [
    np.zeros(5),             # Initial point [0, 0, 0, 0, 0]
    np.ones(5),              # Initial point [1, 1, 1, 1, 1]
    np.full(5, 5),           # Initial point [5, 5, 5, 5, 5]
    np.random.rand(5),       # Random initial point
    np.array([10, -10, 10, -10, 10])  # Mixed positive/negative initial point
]

# Set up the subplots grid (2 rows x 3 columns to fit 5 initial values)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Loop through the initial values and run Newton's method for each
for idx, x_init in enumerate(initial_values):
    # Run Newton's method for each initial value
    x_0, f_values = newton_method(f2, max_iter=100, x_0=x_init)
    
    # Select the correct subplot (flatten axs to handle 2x3 grid)
    ax = axs.flatten()[idx]

    # Plot the convergence of function values over iterations
    ax.plot(np.arange(len(f_values)), f_values)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Function Value')
    ax.set_title(f'Convergence from Initial Value {x_init}')

    # Print information for each initial value
    print(f'Optimal Solution for initial value {x_init} after 100 iterations = {x_0}')
    print(f'Function Value at Optimal Solution = {f_values[-1]}')

# Hide the empty 6th subplot
fig.delaxes(axs[1, 2])

# Adjust layout and display the subplots
plt.tight_layout()
plt.savefig('Newton_Method_Convergence_5_Initial_Values.png')
plt.show()



# Question 3.1
x_0,f_values = constant_gradient_descent(1e-1,f3,x_0 = np.ones(5))
print(f'Optimal Solution after 100 iterations= {x_0}')
plt.plot(np.arange(len(f_values)),f_values)
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.title('Convergence of Constant Step Gradient Descent')
print(np.linalg.norm(f3(x_0,srn,1)))
print(f3(x_0,srn,1))
print('The Best function value over all the iterations is:',round(np.min(f_values),4))
# plt.savefig('grad_3_1.png')
plt.show()


# Question 3.3
print(f'Newton Mehtod Question 3.2')
x_0, f_values = newton_method(f3, x_0=np.ones(5))
print(np.linalg.norm(f3(x_0, srn, 1))) # Check if the norm of the gradient is close to 0
print(f3(x_0, srn, 1)) # Check if the gradient is close to 0Vector
print(f_values)
print(f'Optimal Solution after 100 iterations = {x_0}')
plt.plot(np.arange(10), f_values[0:10])
plt.ylabel('Function Value')
plt.xlabel('Iterations')
plt.title('Convergence of Newton Method')
plt.show()


# Question 3.4
K_values = [35, 40, 55, 70]
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Creating a 2x2 grid for subplots
srn = 22219  # Assuming your serial number is passed here

for idx, K in enumerate(K_values):
    # Gradient Descent for K iterations
    x_0_g_desc, f_values = constant_gradient_descent(1e-2, f3, max_iter=K, x_0=np.ones(5))
    # Newton's Method for the remaining iterations
    x_0_final, f_values_newton = newton_method(f3, max_iter=100-K, x_0=x_0_g_desc)

    # Select the correct subplot
    ax = axs[idx // 2, idx % 2]

    # Plot for gradient descent and Newton's method
    ax.plot(np.arange(K+1), f_values, label='Gradient Descent', color='blue')  # Blue for gradient descent
    ax.plot(np.arange(K, 101), f_values_newton, label="Newton's Method", color='red')  # Red for Newton's method

    # Set plot labels and title
    ax.set_title(f'Convergence with K={K}')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Function Value')
    ax.legend()

    # Print results for each K
    print(f'\nResults for K={K}:')
    print(f'Optimal Solution after 100 iterations = {x_0_final}')
    print(f'Gradient at final point = {f3(x_0_final, srn, 1)}')
    print(f'Norm of gradient = {np.linalg.norm(f3(x_0_final, srn, 1))}')
    print(f'Function value at optimal solution = {round(f3(x_0_final, srn, 0))}')

    # Calculate and print cost
    cost = K + 25 * (100 - K)
    print(f'Total cost for K={K}: {cost}')

# Adjust layout and show the plot
plt.tight_layout()
# plt.savefig('0.01_alpha_hybrid.png')
plt.show()


#Question 4.2
# Rank One Update
print('Rank One Update')
x_0, f_values_rank_one = rank_one_update(f2,  x_0 =  np.zeros(5))
print(f'Optimal Solution after {len(f_values_rank_one)-1} iterations of Rank One Update = {x_0}')
print(f'Function Value at optimal solution: {f_values_rank_one[-1]}')
print('Gradient Norm:', np.linalg.norm(f2(x_0, srn, 1)))

# Constant Gradient Descent
print('Constant Gradient Descent')
x_0, f_values_cgd = constant_gradient_descent(1e-1, f2, x_0=np.zeros(5))
print(f'Optimal Solution after {len(f_values_cgd)-1} iterations of Constant Gradient Descent = {x_0}')
print(f'Function Value at optimal solution: {f_values_cgd[-1]}')
print('Gradient Norm:', np.linalg.norm(f2(x_0, srn, 1)))

# Scalar Hessian Approximation (Quasi-Newton)
print('Scalar Hessian Update')
x_0, f_values_scalar = quasi_newton_scalar(f2, x_0=np.zeros(5))
print(f'Optimal Solution after {len(f_values_scalar)-1} iterations of Scalar Hessian Update = {x_0}')
print(f'Function Value at optimal solution: {f_values_scalar[-1]}')
print('Gradient Norm:', np.linalg.norm(f2(x_0, srn, 1)))

# Plotting all three convergence plots as subplots in a single figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Subplot 1: Rank One Update
axs[0].plot(np.arange(len(f_values_rank_one)), f_values_rank_one)
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Function Value')
axs[0].set_title('Convergence of Rank One Update')

# Subplot 2: Constant Gradient Descent
axs[1].plot(np.arange(len(f_values_cgd)), f_values_cgd)
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Function Value')
axs[1].set_title('Convergence of Constant Gradient Descent')

# Subplot 3: Scalar Hessian Approximation
axs[2].plot(np.arange(len(f_values_scalar)), f_values_scalar)
axs[2].set_xlabel('Iterations')
axs[2].set_ylabel('Function Value')
axs[2].set_title('Convergence of Scalar Hessian Update')

# Display all plots
plt.tight_layout()
# plt.savefig('4.2_Convergence_Plots.png')
plt.show()



