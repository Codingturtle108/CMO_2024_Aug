import numpy as np  # Importing the numpy library
import matplotlib.pyplot as plt # Importing the matplotlib library
import cvxpy as cp  # Importing the cvxpy library
import pandas as pd # Importing the pandas library


# Question 1 Least norm Solution

# Defining the A matrix and b vector
A = np.array([[2, -4, 2, -14], 
              [-1, 2, -2, 11]
            ])
b = np.array([10, -6])
# A is not full Row Rank,We can drop  a row from A to make it full row rank or use pseudo inverse
# We can drop last row without changing the solution as it is a linear multiple of the first row , i.e b1 = -2*b3, R1 = -2*R3.

def Projection_operator(A, b, z):  # Project on X = {x: Ax = b}
    AA_T_inv = np.linalg.inv(A @ A.T)  # Compute pseudo-inverse of A*A^T
    return z - (z @ A.T) @ AA_T_inv @ A + A.T @ (AA_T_inv @ b)


# Calculating x* Using KKT Conditions 
x_star = A.T@np.linalg.inv(A @ A.T) @ b
print(f'The Optimal Solution using KKT Conditions is: {x_star}')    
print(x_star)

def Projected_Gradient_Descent(A, b, z0, alpha, num_iters):
    z = z0  # Initialize z
    x_vals = [z]  # Store the x values
    for t in range(num_iters):
        z = Projection_operator(A, b, z - alpha*z)  # Update z
        x_vals.append(z)  # Store the updated z
    return x_vals

# Run Projected Gradient Descent for different values of alpha
z_0 = 10*np.random.rand(4)  # Random initial point
#Initialize the plot
plt.figure(figsize=(8, 6))
for alpha in [1, 1e-1, 1e-2, 1e-3, 1e-4]:  # Different values of alpha (step size)
    if alpha in [1, 1e-1]:
        num_iters = 100
    elif alpha == 1e-2:
        num_iters = 1000
    else:
        num_iters = 100000  # Number of iterations for PGD
    
    x_vals = Projected_Gradient_Descent(A, b, z_0, alpha, num_iters)
    x_vals = np.array(x_vals) - x_star  # Compute x_t - x*
    x_norms = np.linalg.norm(x_vals, axis=1)  # Compute ||x_t - x*||
    
    # Plot ||x_t - x*|| vs. Iterations for each alpha
    plt.plot(x_norms, label=f'alpha = {alpha}')

# Finalize the plot
plt.xlabel('Iterations')
plt.ylabel('||x_t - x*||')
plt.title('Convergence of Projected Gradient Descent')
plt.legend()
plt.grid()
plt.savefig('Projected_Gradient_Descent.png')
plt.show()
# Here we can seet that for alpha less than 2/L here L = 1 since Quadratic form is ||x||^2, spectral norm i.e L for quadratic functions is 1

# Question 2  Support Vector Machine
# Defining the data
# Question 2.1

X = pd.read_csv('Data/Data.csv',header = None).to_numpy()
Y = pd.read_csv('Data/Labels.csv',header = None).to_numpy().flatten()

# Define dimensions
n = X.shape[1]  # Number of features
N = X.shape[0]  # Number of data points

# Define variables for the primal problem
w = cp.Variable(n)
b = cp.Variable()

# Define the objective function for the primal problem
objective = cp.Minimize(0.5 * cp.norm(w, 2)**2)

# Define constraints for the primal problem
constraints = [Y[i] * (X[i, :] @ w + b) >= 1 for i in range(N)]

# Set up and solve the primal optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Retrieve the optimal values of w and b
w_opt = w.value
b_opt = b.value
Gamma1 = 0
Gamma2 = 0 
for i, constr in enumerate(constraints):
    if Y[i] > 0:
        Gamma1 += constr.dual_value
    if Y[i] < 0:
        Gamma2 += constr.dual_value
print("Gamma1:",Gamma1)
print("Gamma2:",Gamma2)
print("Optimal weight vector (w):", w_opt)
print("Optimal bias (b):", b_opt)
print("Value of Primal Objective:", problem.value)  # Value of the primal objective

# Plot the data and the decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', label='Class +1')
plt.scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', label='Class -1')

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
xx = np.linspace(x_min, x_max, 100)
yy = -(w_opt[0] * xx + b_opt) / w_opt[1]
plt.plot(xx, yy, 'k--', label='Decision Boundary')

# Plot margins
margin = 1 / np.linalg.norm(w_opt)
yy_margin_pos = yy + margin
yy_margin_neg = yy - margin
plt.plot(xx, yy_margin_pos, 'k:', label='Margin')
plt.plot(xx, yy_margin_neg, 'k:')

plt.xlim(x_min, x_max)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Support Vector Machine Decision Boundary (Primal)")
plt.savefig('SVM_Primal.png')
plt.show()


# Question 2.4 Solving the Dual Problem

# Define the variables for the dual problem
# Load data
X = pd.read_csv('Data/Data.csv', header=None).to_numpy()
Y = pd.read_csv('Data/Labels.csv', header=None).to_numpy().flatten()
n = X.shape[0]

# Compute Q-matrix
Q = (Y[:, None] * Y[None, :]) * (X @ X.T)
L = np.max(abs(np.linalg.eigvals(Q)))
print(2/L)

# Projection function
def Projection_Y(x): # Projection on Y.T.x  = 0 and x >= 0
   return np.maximum(x-Y*np.dot(Y,x)/(np.linalg.norm(Y)**2), 0)


# Dual value function
def Dual_Value(x, Q):
    return -0.5 * x.T @ Q @ x + np.sum(x)

# Projected Gradient Descent
def Projected_Gradient_Descent(Q, x0, alpha, num_iters, tolerance=1e-6):
    x = x0
    lamda_vals = [x]
    # f_vals = [Dual_Value(x, Q)]
    for t in range(num_iters):
        grad = Q @ x - np.ones(n)
        x_new = Projection_Y(x - alpha * grad)
        lamda_vals.append(x_new)
        # f_vals.append(Dual_Value(x_new, Q))
        
        # Stopping criterion
        if np.linalg.norm(grad) < tolerance:
            break
        x = x_new
    return lamda_vals #, f_vals
## Active Set Method




# Parameters
x_0 = np.random.rand(n)
alpha = 1e-3
num_iters = 10000000

# Run PGD
lamda_vals = Projected_Gradient_Descent(Q, x_0, alpha, num_iters)

# Output results
print('The Optimal Value of the Dual Problem is:', Dual_Value(lamda_vals[-1], Q))
print('The Optimal value is attained at x =', lamda_vals[-1])
w = np.sum([lamda_vals[-1][i] * Y[i] * X[i, :] for i in range(n)], axis=0)  # Calculate w
b = np.mean([Y[i] - np.dot(w, X[i, :]) for i in range(n) if lamda_vals[-1][i] > 0]) # Calculate b (Taking average of all the points where lamda > 0 to take into account Numerical Errors)
print('The Optimal value of w is:', w)
# Verify 2.3 # Calculate Value of Gamma
Gamma1 = np.sum([Y[i]*lamda_vals[-1][i] for i in range(n) for i in range(n) if Y[i] > 0])
Gamma2 = np.sum([-Y[i]*lamda_vals[-1][i] for i in range(n) for i in range(n) if Y[i] < 0])
print('Gamma:',Gamma1)
print('Gamma:',Gamma2)
 
 # 2.5 Active Constraints
# Find the active constraints
print('lamda values:',lamda_vals[-1])
active_constraints = [i for i in range(n) if lamda_vals[-1][i] > 0] # Active constraints are those where lamda > 1e-6 (Numerical Tolerance)
print(active_constraints)
b = Y[i]- np.dot(w,X[active_constraints[0]])
print(b)
# Find the support vectors
# support_vectors = [X[i,:] for i in active_constraints]
support_vectors = active_constraints
## 2.6 Plotting the data and decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', label='Class +1', marker='o')  # Circles for y = 1
plt.scatter(X[Y == -1, 0], X[Y == -1, 1], color='r', label='Class -1', marker='s')  # Squares for y = -1

# Highlight the support vectors
plt.scatter(X[support_vectors, 0], X[support_vectors, 1], facecolors='none', edgecolors='k', s=100, label='Support Vectors')

# Plot decision boundary: w^T x + b = 0
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
xx = np.linspace(x_min, x_max, 100)
yy = -(w[0] * xx + b) / w[1]
plt.plot(xx, yy, 'k--', label='Decision Boundary')

# Plot margins: w^T x + b = ±1
yy_margin_pos = -(w[0] * xx + b - 1) / w[1]
yy_margin_neg = -(w[0] * xx + b + 1) / w[1]
plt.plot(xx, yy_margin_pos, 'k:', label='Positive Margin')
plt.plot(xx, yy_margin_neg, 'k:', label='Negative Margin')

# Add labels and legend
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary with Support Vectors')
plt.legend()
plt.grid()
plt.savefig('SVM_Dual.png')
plt.show()

