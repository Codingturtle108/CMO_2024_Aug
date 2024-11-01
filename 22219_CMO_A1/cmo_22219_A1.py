from CMO_A1 import f1,f2,f3,f4
import numpy as np
import matplotlib.pyplot as plt
srn  = 22219
#Question 1 Checking for convexity,Strong Convexity and Coercivity
# 1.1 Convexity and Strict Convexity
def isConvex(func, interval):
    x_values = np.linspace(interval[0], interval[1], 100)
    
    for i in range(len(x_values) - 2):
        x1 = x_values[i]
        x2 = x_values[i + 2]
        midpoint = (x1 + x2) / 2
        # Check Jensen's inequality
        if func(srn, midpoint) > 0.5 * (func(srn, x1) + func(srn, x2)):
            return False
    return True

interval = [-2,2]
is_convex_f1 = isConvex(f1,interval)
print(f'f1 is convex: {is_convex_f1}')
is_convex_f2 = isConvex(f2,interval)
print(f'f2 is convex: {is_convex_f2}')
#Strong Convexity
def isStrictlyConvex(func, interval):
    x_values = np.linspace(interval[0], interval[1], 300)
    
    for i in range(len(x_values) - 2):
        x1 = x_values[i]
        x2 = x_values[i + 2]
        midpoint = (x1 + x2) / 2
        # Check Jensen's inequality
        if func(srn, midpoint) >= 0.5 * (func(srn, x1) + func(srn, x2)):
            return False
    return True

interval = [-2,2]
is_convex_f1 = isStrictlyConvex(f1,interval)
print(f'f1 is Strictly convex: {is_convex_f1}')
is_convex_f2 = isStrictlyConvex(f2,interval)
print(f'f2 is  Strictly convex: {is_convex_f2}')
# finding x* and f(x*)
# Grid search to find the minimum
def grid_search_minimum(f, interval, num_points=300):
    x_values = np.linspace(interval[0], interval[1], num_points)
    f_values = [f(srn, x) for x in x_values]
    
    # Find the index of the minimum function value
    min_index = np.argmin(f_values)
    x_star = x_values[min_index]
    f_x_star = f_values[min_index]
    minimas  = [i  for i in f_values if i == f_x_star] 
    if len(minimas) == 1:
        return x_star, f_x_star,True
    
    return x_star, f_x_star,False

interval = [-2, 2] # Interval in which to search for the minimum
x1_star, f_x1_star,uniquenessf1 = grid_search_minimum(f1, interval)

print("x* (point of minima of f1):", x1_star)
print("f(x*) (function value at f1(x*)):", f_x1_star)
print(f'Minima is Unique {uniquenessf1}')
x2_star,f_x2_star,uniquenessf2 = grid_search_minimum(f2,interval)
print("x* (point of minima of f2):", x2_star)
print("f(x*) (function value at f2(x*)):", f_x2_star)
print(f'Minima is Unique {uniquenessf2}')

# 1.2 Coercivity
# f3 is a quartic polynomial we need to check for coercivity and find stationary points
def find_coeff(f):
      #a,b,c,d,e are  coefficients of x^4,x^3,x^2,x^1,x^0
    X = np.array([f(srn,i) for i in range(0,5)]) #A.(e,d,c,b,a) =X where A is vandermonde matrix 
    A = np.vander([0,1,2,3,4],increasing=True)
    coeff = np.linalg.solve(A,X)
    return coeff
def is_coercive(f): #We need to find value of the coeff of x^4 if positive then it is coercive
    coeff = find_coeff(f)
    if coeff[-1] > 0:
        return True
    else:
        return False
print(find_coeff(f3))
print(is_coercive(f3))
#Stationary points
#Finding Stationary points of quartic function f3
def stationary_points(f):
    function_dict = {"roots":[],"minima":[],"maxima":[],"local_maxima":[],"local_minima":[]}
    coeff = find_coeff(f)
       

    #f'(x) = 4*coeff[-1]*x^3 + 3*coeff[-2]*x^2 + 2*coeff[-3]*x + coeff[-4] = 0
    #Let y = x^2
    #4*coeff[0-1]*y^3 + 3*coeff[1]*y^2 + 2*coeff[2]*y + coeff[3] = 0
    y = np.roots([4*coeff[-1],3*coeff[-2],2*coeff[-3],coeff[-4]])
    #f''(x) = 12*coeff[-1]*x^2 + 6*coeff[-2]*x + 2*coeff[-3]
    f_second_grad = lambda y : 12*coeff[-1]*(y**2) + 6*coeff[-2]*y + 2*coeff[-3]
    for i in y :
        if f_second_grad(i) > 0:
            function_dict["local_minima"].append(i)
        else:
            function_dict["local_maxima"].append(i)
    if coeff[-1] > 0:
        function_dict["maxima"] = ['inf']
        minima  = np.argmin([f3(srn,i) for i  in function_dict['local_minima']])
        function_dict['minima'] = np.array(function_dict["local_minima"][minima])
    else:
        function_dict["minima"] = ['-inf']
        maxima  = np.argmax([f3(srn,i) for i  in function_dict['local_maxima']])
        function_dict['maxima'] = np.array(function_dict["local_maxima"][maxima])
    function_dict["roots"] = np.roots([coeff[-1],coeff[-2],coeff[-3],coeff[-4],coeff[-5]]) 
    return function_dict
print(stationary_points(f3))

############################################################################################################################################################


# Question 2  Gradient Descent
# Helper Function for Plotting Graphs
def Plotter_Q2(x_t,grad_norms,iter_values,iter_fvalues,Save_path=False):
    ratio_list = np.zeros(max_iter)
    ratio_norms = np.zeros(max_iter)
    diff_list = iter_fvalues - iter_fvalues[-1] # f(x) - f(x_t)

    for i in range(1, max_iter+1):
        ratio_list[i-1] = (iter_fvalues[i] - iter_fvalues[-1]) / (iter_fvalues[i - 1] - iter_fvalues[-1])
        ratio_norms[i-1] = (np.linalg.norm(iter_values[i] - iter_values[-1]) / np.linalg.norm(iter_values[i - 1] - iter_values[-1])) ** 2
    
    plt.figure(figsize=(12, 10))

    # Plot 1: f(x) - f(x_t)
    plt.subplot(2, 2, 1)
    plt.plot(range(max_iter), diff_list[:max_iter])
    plt.xlabel('Iterations')
    plt.ylabel('f(x) - f(x_t)')
    plt.title('Difference: f(x) - f(x_t)')

    # Plot 2: Ratio of (f(x) - f(x_t)) / (f(x_prev) - f(x_t))
    plt.subplot(2, 2, 2)
    plt.plot(range(max_iter), ratio_list[:max_iter])
    plt.xlabel('Iterations')
    plt.ylabel('(f(x) - f(x_t)) / (f(x_prev) - f(x_t))')
    plt.title('Convergence Ratio of f(x)')

    # Plot 3: Gradient norm ||grad(f(x))||
    plt.subplot(2, 2, 3)
    plt.plot(range(max_iter), grad_norms[:max_iter])
    plt.xlabel('Iterations')
    plt.ylabel('||grad(f(x))||')
    plt.title('Gradient Norms')

    # Plot 4: Ratio of ||x - x_t||^2 / ||x_prev - x_t||^2
    plt.subplot(2, 2, 4)
    plt.plot(range(max_iter), ratio_norms[:max_iter])
    plt.xlabel('Iterations')
    plt.ylabel('||x - x_t||^2 / ||x_prev - x_t||^2')
    plt.title('Convergence Ratio of x')
    plt.tight_layout()
    if Save_path:
        plt.savefig(Save_path)
    plt.show()

    
# 2.1 ConstantGradientDescent
def ConstantGradientDescent(alpha, initial_x=np.zeros(5), max_iter=10000):
    gradient_norms = np.zeros(max_iter + 1)
    iter_fvalues = np.zeros(max_iter + 1)
    iter_values = np.zeros((max_iter + 1, len(initial_x)))  # Store the full x vectors at each iteration
    
    # Initialize values
    gradient_norms[0] = np.linalg.norm(f4(srn,initial_x)[1])
    iter_values[0] = initial_x
    iter_fvalues[0] = f4(srn, initial_x)[0]
    
    x_0 = initial_x
    iter_count = 0
    
    while iter_count < max_iter:
        _, grad = f4(srn, x_0)
        x_1 = x_0 - alpha * grad
        
        iter_count += 1
        
        # Update norms, values, and gradients
        gradient_norms[iter_count] = np.linalg.norm(grad)
        iter_fvalues[iter_count] = f4(srn, x_1)[0]
        iter_values[iter_count] = x_1
        
        x_0 = x_1

    return x_1, gradient_norms, iter_fvalues, iter_values

# Perform Gradient Descent
alpha = 1e-5
max_iter = 13000
x_t, grad_norms, iter_fvalues, iter_values = ConstantGradientDescent(alpha, np.zeros(5), max_iter)

plot_path = 'Q2ConstantGradientDescent.png'
Plotter_Q2(x_t,grad_norms,iter_values,iter_fvalues,plot_path)
print(f'The Value of x_t is {x_t} and f(x_t) is {f4(srn,x_t)[0]} and gradient norm is {grad_norms[-1]}')

############################################################################################################################################################


# 2.2 Diminsihing Gradient Descent
def Diminishing_Gradient_Descent(alpha, initial_x=np.zeros(5), max_iter=10000):
    gradient_norms = np.zeros(max_iter + 1)
    iter_fvalues = np.zeros(max_iter + 1)
    iter_values = np.zeros((max_iter + 1, len(initial_x)))  # Store the full x vectors at each iteration
    
    # Initialize values
    gradient_norms[0] = np.linalg.norm(f4(srn,initial_x)[1])
    iter_values[0] = initial_x
    iter_fvalues[0] = f4(srn, initial_x)[0]
    
    x_0 = initial_x
    iter_count = 0
    
    while iter_count < max_iter:
        _, grad = f4(srn, x_0)
        x_1 = x_0 - alpha * grad/(iter_count+1)
        
        iter_count += 1
        
        # Update norms, values, and gradients
        gradient_norms[iter_count] = np.linalg.norm(grad)
        iter_fvalues[iter_count] = f4(srn, x_1)[0]
        iter_values[iter_count] = x_1
        
        x_0 = x_1

    return x_1, gradient_norms, iter_fvalues, iter_values

# Perform Gradient Descent
alpha = 1e-3
max_iter = 14000
x_t, grad_norms, iter_fvalues, iter_values = Diminishing_Gradient_Descent(alpha, np.zeros(5),max_iter)
plot_path = 'Q2DiminishingGradientDescent.png'
Plotter_Q2(x_t,grad_norms,iter_values,iter_fvalues,False)  #Plotting 

print(f'The Value of x_t is {x_t} and f(x_t) is {f4(srn,x_t)[0]} and gradient norm is {grad_norms[-1]}')



############################################################################################################################################################


#2.3 Inexact Line Search
def Inexact_Line_Search(c1,c2,gamma, max_iter=10000):
    gamma = 0.5
    initial_x = np.zeros(5)
    gradient_norms = np.zeros(max_iter + 1)
    iter_fvalues = np.zeros(max_iter + 1)
    iter_values = np.zeros((max_iter + 1, len(initial_x)))  # Store the full x vectors at each iteration
    
    # Initialize values
    gradient_norms[0] = np.linalg.norm(f4(srn,initial_x)[1])
    iter_values[0] = initial_x
    iter_fvalues[0] = f4(srn, initial_x)[0]
    
    x_0 = initial_x
    iter_count = 0
    
    while iter_count < max_iter:
        alpha = 1
        _, grad = f4(srn, x_0)
        while f4(srn, x_0 - alpha * grad)[0] > f4(srn, x_0)[0] - c1 * alpha * np.dot(grad, grad) and np.dot(grad, f4(srn, x_0 - alpha * grad)[1]) < c2 * np.dot(grad, grad):
            alpha = gamma * alpha
        x_1 = x_0 - alpha * grad
        
        iter_count += 1
        
        # Update norms, values, and gradients
        gradient_norms[iter_count] = np.linalg.norm(grad)
        iter_fvalues[iter_count] = f4(srn, x_1)[0]
        iter_values[iter_count] = x_1
        
        x_0 = x_1

    return x_1, gradient_norms, iter_fvalues, iter_values

# Perform Gradient Descent
alpha = 1e-3
max_iter = 1000
x_t, grad_norms, iter_fvalues, iter_values = Inexact_Line_Search(0.1,0.9,0.5,max_iter)
plot_path = 'Q2InexactLineSearch.png'
Plotter_Q2(x_t,grad_norms,iter_values,iter_fvalues,False)  #Plotting    
print(f'The Value of x_t is {x_t} and f(x_t) is {f4(srn,x_t)[0]} and gradient norm is {grad_norms[-1]}')

#2.4 Exact Line Search

# Quad  = 0.5x^T * A * x + b^T * x + c 
#Finding values of b,c quad(0) = c and quad'(0) = b
constant_c = f4(srn,np.zeros(5))[0]
constant_b = f4(srn,np.zeros(5))[1]
print(constant_c,constant_b)

############################################################################################################################################################


def Exact_Line_Search( max_iter=10000):
    gamma = 0.5
    initial_x = np.zeros(5)
    gradient_norms = np.zeros(max_iter + 1)
    iter_fvalues = np.zeros(max_iter + 1)
    iter_values = np.zeros((max_iter + 1, len(initial_x)))  # Store the full x vectors at each iteration
    
    # Initialize values
    gradient_norms[0] = np.linalg.norm(f4(srn,initial_x)[1])
    iter_values[0] = initial_x
    iter_fvalues[0] = f4(srn, initial_x)[0]
    
    x_0 = initial_x
    iter_count = 0
    
    while iter_count < max_iter:
        _, grad = f4(srn, x_0)
        temp,_ = f4(srn,grad)
        delta = 2*(temp-np.ones(5).dot(grad)) # pk.T*H.pk = 2*(f(x+pk)-f(x)-grad(f(x)).T*pk)
        alpha = np.dot(grad, grad) / delta
        x_1 = x_0 - alpha * grad
        iter_count += 1

        # Update norms, values, and gradients
        gradient_norms[iter_count] = np.linalg.norm(grad)
        iter_fvalues[iter_count] = f4(srn, x_1)[0]
        iter_values[iter_count] = x_1
        
        x_0 = x_1

    return x_1, gradient_norms, iter_fvalues, iter_values

# Perform Gradient Descent
alpha = 1e-3
max_iter = 1000
x_t, grad_norms, iter_fvalues, iter_values = Exact_Line_Search(max_iter)

plot_path = 'Q2ExactLineSearch.png'
Plotter_Q2(x_t,grad_norms,iter_values,iter_fvalues,False)  #Plotting
print(f'The Value of x_t is {x_t} and f(x_t) is {f4(srn,x_t)[0]} and gradient norm is {grad_norms[-1]}')

############################################################################################################################################################


#Question 3 Pertubed Gradient Descent
 # 3.3 Contour Plot
def f(x, y):
    return np.exp(x * y)

# Generate grid of points
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the contour plot using matplotlib
plt.figure(figsize=(6,6))
contour = plt.contour(X, Y, Z, levels=20, cmap="viridis")
plt.colorbar(contour)
plt.title(r'Contour plot of $f(x,y) = e^{xy}$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

############################################################################################################################################################

# Helper Function for Plotting 
def PlotterQ3GD(trajectory,f_values,Save_path = False):
    x = np.linspace(-1, 1, 400)
    y = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(X * Y)   # Compute the function values

    # Create the contour plot using matplotlib
    plt.figure(figsize=(10, 5))

    # Plot the trajectory on the contour plot
    plt.subplot(1, 2, 1)  # Subplot for contour + trajectory
    contour = plt.contour(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(contour)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', label='Gradient Descent Trajectory')
    plt.scatter(starting_point[0], starting_point[1], color='red', label='Start', zorder=5)  # Show starting point
    plt.title(r'Trajectory of Gradient Descent on $f(x,y) = e^{xy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    # Plotting function values over iterations
    plt.subplot(1, 2, 2)  # Subplot for function values over iterations
    plt.plot(range(iterations+1), f_values, 'b-')
    plt.xlabel('Iterations')
    plt.ylabel('f(x, y)')
    plt.title('Function values over iterations')
    plt.grid(True)

    # Save the plot
    plt.tight_layout()  # Adjust layout for better spacing
    if Save_path:
        plt.savefig(Save_path) #save the plot
    plt.show()
    
    

# 3.4 Constant Gradient Descent
def gradient_descent(alpha, iterations, initial_x, initial_y):
    x_0, y_0 = initial_x, initial_y
    trajectory = np.zeros((iterations + 1, 2))  # Store 2D points (x, y)
    iter_count = 0
    trajectory[0] = np.array([x_0, y_0])  # Initial point
    f_values = np.zeros(iterations + 1) # Store function values
    f_values[0] = np.exp(x_0*y_0)  # Initial value of f(x, y)
    while iter_count < iterations:
        grad_x, grad_y = y_0 * np.exp(x_0 * y_0), x_0 * np.exp(x_0 * y_0)
        x_1 = x_0 - alpha * grad_x
        y_1 = y_0 - alpha * grad_y
        x_0, y_0 = x_1, y_1
        iter_count += 1
        trajectory[iter_count] = np.array([x_0, y_0])  # Store the updated point
        f_values[iter_count] = np.exp(x_0 * y_0) # Store the updated function value

    return x_0, y_0, trajectory,f_values  # Return final point and trajectory history
initial_x, initial_y = 2,2
starting_point = np.array([initial_x, initial_y])
alpha = 1e-3
iterations = 10000

final_x, final_y, trajectory, f_values = gradient_descent(alpha, iterations, initial_x, initial_y)

plot_path = "Q3_4_plot.png"
PlotterQ3GD(trajectory,f_values,False)

############################################################################################################################################################
 # 3.5 Diminishing Gradient Descent
def diminishing_gradient_descent(alpha, iterations, initial_x, initial_y):
    x_0, y_0 = initial_x, initial_y
    trajectory = np.zeros((iterations + 1, 2))  # Store 2D points (x, y)
    iter_count = 0
    trajectory[0] = np.array([x_0, y_0])  # Initial point
    f_values = np.zeros(iterations + 1) # Store function values
    f_values[0] = np.exp(x_0*y_0)  # Initial value of f(x, y)
    while iter_count < iterations:
        grad_x, grad_y = y_0 * np.exp(x_0 * y_0), x_0 * np.exp(x_0 * y_0)
        x_1 = x_0 - alpha * grad_x/(iter_count+1)
        y_1 = y_0 - alpha * grad_y/(iter_count+1)
        x_0, y_0 = x_1, y_1
        iter_count += 1
        trajectory[iter_count] = np.array([x_0, y_0])  # Store the updated point
        f_values[iter_count] = np.exp(x_0 * y_0) # Store the updated function value

    return x_0, y_0, trajectory,f_values  # Return final point and trajectory history
    
initial_x, initial_y = 1,1
starting_point = np.array([initial_x, initial_y])
alpha = 0.1
iterations = 10000

# Assuming gradient_descent is defined
final_x, final_y, trajectory, f_values =diminishing_gradient_descent(alpha, iterations, initial_x, initial_y)
plot_path = "Q3_5_plot.png"
PlotterQ3GD(trajectory,f_values,False)
print(trajectory[-1],f_values[-1])

############################################################################################################################################################
# Helper Function for Plotting Trajectories, Expected Function Values, and Individual Trajectories
def Plotter_Q3_Stochastic_GD(algo,alpha,iterations,n,Save_path=False): # n is the number of Sample Trajectories # algo is the optimizing algorithm
    trajectory_exp = np.zeros((iterations + 1, 2))
    f_trajectory_exp = np.zeros(iterations + 1)

    # Running the perturbed gradient descent multiple times and accumulating the trajectories
    for i in range(n):
        _, _, trajectory, f_trajectory = algo(alpha, iterations, initial_x, initial_y)
        trajectory_exp += trajectory
        f_trajectory_exp += f_trajectory

    trajectory_exp /= n  # Compute the average trajectory
    f_trajectory_exp /= n  # Compute the average function value

    # Generate grid for contour plot
    x = np.linspace(-1, 1, 400)
    y = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(X * Y)  # Compute the function values

    # Create a figure with 3 subplots (1 row and 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Subplot 1: Contour plot and average trajectory
    contour1 = ax1.contour(X, Y, Z, levels=20, cmap="viridis")
    fig.colorbar(contour1, ax=ax1)  # Add colorbar to the contour plot
    ax1.plot(trajectory_exp[:, 0], trajectory_exp[:, 1], 'ro-', label='Average Gradient Descent Trajectory')
    ax1.scatter(initial_x, initial_y, color='blue', label='Start', zorder=5)
    ax1.set_title(r'Trajectory of Gradient Descent on $f(x,y) = e^{xy}$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Expected function values over iterations
    ax2.plot(range(iterations + 1), f_trajectory_exp[:iterations + 1], 'ro-', label='Expected f Values')
    ax2.set_title('Expected f Values over Iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('f values')
    ax2.legend()
    ax2.grid(True)

    # Subplot 3: Individual trajectory from the last run
    contour2 = ax3.contour(X, Y, Z, levels=20, cmap="viridis")
    fig.colorbar(contour2, ax=ax3)  # Add colorbar to the contour plot
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'bo-', label='Single Run Trajectory')
    ax3.scatter(initial_x, initial_y, color='red', label='Start', zorder=5)  # Start point
    ax3.scatter(trajectory[-1][0], trajectory[-1][1], color='green', label='End', zorder=5)  # End point
    ax3.set_title('Trajectory of a Single Gradient Descent Run')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend()
    ax3.grid(True)

    # Adjust layout
    plt.tight_layout()
    if Save_path:
        plt.savefig(Save_path)  # Save the plot
    # Show the plot
    plt.show() 

 # 3.6  Pertubed Gradient Descent fixed step and fixed variance
def pertubed_gradient_descent_fixed_step(alpha, iterations, initial_x, initial_y):
    x_0, y_0 = initial_x, initial_y
    trajectory = np.zeros((iterations + 1, 2))  # Store 2D points (x, y)
    f_trajectory = np.zeros(iterations + 1)  # Store function values
    iter_count = 0
    trajectory[0] = np.array([x_0, y_0])  # Initial point
    f_trajectory[0] = np.exp(x_0 * y_0)
    random_perturbation = np.random.multivariate_normal(mean =np.zeros(2), cov = np.eye(2)*0.01,size =iterations)
    
    while iter_count < iterations:
        grad_x, grad_y = y_0 * np.exp(x_0 * y_0), x_0 * np.exp(x_0 * y_0)
        # random_perturbation = np.random.multivariate_normal(mean =np.zeros(2), cov = np.eye(2)*0.01,size =1)
        x_1 = x_0 - alpha * grad_x + random_perturbation[iter_count][0]
        y_1 = y_0 - alpha * grad_y + random_perturbation[iter_count][1]
        x_0, y_0 = x_1, y_1
        iter_count += 1
        trajectory[iter_count] = np.array([x_0, y_0])  # Store the updated point
        f_trajectory[iter_count] = np.exp(x_0 * y_0) # Store the function value

    return x_0, y_0, trajectory,f_trajectory  # Return final point and trajectory history    


# Parameters
alpha = 1e-2
initial_x, initial_y = 1, 1
n = 100 # Number of runs
iterations = 10000  # Number of iterations per run\
plot_path = "Q3_6_Plot.png"
Plotter_Q3_Stochastic_GD(pertubed_gradient_descent_fixed_step,alpha,iterations,n,False) # Plot the trajectory and function  


############################################################################################################################################################
#3.7 Perturbed Gradient Descent with Fixed Step Size and decreasing variance

def pertubed_gradient_descent_fixed_step_decreasing_var(alpha, iterations, initial_x, initial_y):
    x_0, y_0 = initial_x, initial_y
    trajectory = np.zeros((iterations + 1, 2))  # Store 2D points (x, y)
    iter_count = 0
    trajectory[0] = np.array([x_0, y_0])  # Initial point
    f_trajectory = np.zeros(iterations + 1)  # Store function values
    f_trajectory[0] = np.exp(x_0 * y_0) # Initial function value
    while iter_count < iterations:
        grad_x, grad_y = y_0 * np.exp(x_0 * y_0), x_0 * np.exp(x_0 * y_0)
        random_perturbation = np.random.multivariate_normal(mean =np.zeros(2), cov = np.eye(2)*(0.1/(iter_count+1))**2,size =1)
        x_1 = x_0 - alpha * grad_x + random_perturbation[0][0]
        y_1 = y_0 - alpha * grad_y + random_perturbation[0][1]
        x_0, y_0 = x_1, y_1
        iter_count += 1
        trajectory[iter_count] = np.array([x_0, y_0])  # Store the updated point
        f_trajectory[iter_count] = np.exp(x_0 * y_0)    # Store the function value

    return x_0, y_0, trajectory ,f_trajectory # Return final point and trajectory history


# Parameters
alpha = 1e-2
initial_x, initial_y = 1, 1
n = 1000  # Number of runs
iterations = 100  # Number of iterations per run
plot_path = "Q3_7_Plot.png"
Q3_7 = Plotter_Q3_Stochastic_GD(pertubed_gradient_descent_fixed_step_decreasing_var,alpha,iterations,n,False) # Plot the trajectory and function
############################################################################################################################################################
 # 3.8 Pertubed Gradient Descent with Diminishing Step Size and Fixed Variance
def pertubed_gradient_descent_diminishing_step_and_fixed_var(alpha, iterations, initial_x, initial_y):
    x_0, y_0 = initial_x, initial_y
    trajectory = np.zeros((iterations + 1, 2))  # Store 2D points (x, y)
    iter_count = 0
    trajectory[0] = np.array([x_0, y_0])  # Initial point
    f_trajectory = np.zeros(iterations + 1)  # Store function values
    f_trajectory[0] = np.exp(x_0 * y_0) # Initial function value
    while iter_count < iterations:
        grad_x, grad_y = y_0 * np.exp(x_0 * y_0), x_0 * np.exp(x_0 * y_0)
        random_perturbation = np.random.multivariate_normal(mean =np.zeros(2), cov = np.eye(2)*0.01,size =1)
        x_1 = x_0 - alpha/(iter_count+1) * grad_x + random_perturbation[0][0]
        y_1 = y_0 - alpha/(iter_count+1) * grad_y + random_perturbation[0][1]
        x_0, y_0 = x_1, y_1
        iter_count += 1
        trajectory[iter_count] = np.array([x_0, y_0])  # Store the updated point
        f_trajectory[iter_count] = np.exp(x_0 * y_0)    # Store the function value

    return x_0, y_0, trajectory ,f_trajectory # Return final point and trajectory history


# Parameters
alpha = 1e-1
initial_x, initial_y = 1, 1
n = 1000  # Number of runs
iterations = 100  # Number of iterations per run

plot_path = "Q3_8_Plot.png"
Plotter_Q3_Stochastic_GD(pertubed_gradient_descent_diminishing_step_and_fixed_var,alpha,iterations,n,False) # Plot the trajectory and function

############################################################################################################################################################
# 3.9 Pertubed Gradient Descent with Diminishing Step Size and Decreasing Variance
def pertubed_gradient_descent_diminishing_step_and_decreasing_var(alpha, iterations, initial_x, initial_y):
    x_0, y_0 = initial_x, initial_y
    trajectory = np.zeros((iterations + 1, 2))  # Store 2D points (x, y)
    iter_count = 0
    trajectory[0] = np.array([x_0, y_0])  # Initial point
    f_trajectory = np.zeros(iterations + 1)  # Store function values
    f_trajectory[0] = np.exp(x_0 * y_0) # Initial function value
    while iter_count < iterations:
        grad_x, grad_y = y_0 * np.exp(x_0 * y_0), x_0 * np.exp(x_0 * y_0)
        random_perturbation = np.random.multivariate_normal(mean =np.zeros(2), cov = np.eye(2)*(0.1/(iter_count+1))**2,size =1)
        x_1 = x_0 - alpha/(iter_count+1) * grad_x + random_perturbation[0][0]
        y_1 = y_0 - alpha/(iter_count+1) * grad_y + random_perturbation[0][1]
        x_0, y_0 = x_1, y_1
        iter_count += 1
        trajectory[iter_count] = np.array([x_0, y_0])  # Store the updated point
        f_trajectory[iter_count] = np.exp(x_0 * y_0)    # Store the function value

    return x_0, y_0, trajectory ,f_trajectory # Return final point and trajectory history


# Parameters
alpha = 1e-1
initial_x, initial_y = 1, 1
n = 1000  # Number of runs
iterations = 100  # Number of iterations per run

plot_path = "Q3_9_Plot.png"
Plotter_Q3_Stochastic_GD(pertubed_gradient_descent_diminishing_step_and_decreasing_var,alpha,iterations,n,False) # Plot the trajectory and function

############################################################################################################################################################

# Question 4 
# 4.1 Golden Section Search

def f(x):
    return x * (x - 1) * (x - 3) * (x + 2)

def golden_section_search(a, b, tol=1e-4):
    golden_ratio = (1 + np.sqrt(5)) / 2
    iteration_count = 0  # Initialize the iteration counter
    f_avalues = [f(a)]
    f_bvalues = [f(b)]
    diff_list = [b-a]

    while (b - a) > tol:
        iteration_count += 1  # Increment iteration counter
        x1 = b - (b - a) / golden_ratio
        x2 = a + (b - a) / golden_ratio
        
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        
        f_avalues.append(f(a))
        f_bvalues.append(f(b))
        diff_list.append(b-a)
        
    return [a,b], iteration_count, f_avalues, f_bvalues, diff_list

# Run Golden Section Search
result, iterations, f_avalues, f_bvalues, diff_list = golden_section_search(1, 3)
print(f"Golden Section Search result: x* in interval {result}, iterations = {iterations}")

# Calculate the ratio list
ratio_list = np.zeros(iterations)
for i in range(1, iterations + 1):
    ratio_list[i - 1] = (f_bvalues[i] - f_avalues[i]) / (f_bvalues[i - 1] - f_avalues[i - 1])

# Plotting
plt.figure(figsize=(10, 8))

# Plot f(a_t) vs iterations
plt.subplot(2, 2, 1)
plt.plot(np.arange(0, iterations + 1), f_avalues[0:iterations + 1], 'ro-', label='f(a_t) vs iterations')
plt.xlabel('iterations')
plt.ylabel('f(a_t)')
plt.legend()

# Plot f(b_t) vs iterations
plt.subplot(2, 2, 2)
plt.plot(np.arange(0, iterations + 1), f_bvalues[0:iterations + 1], 'bo-', label='f(b_t) vs iterations')
plt.xlabel('iterations')
plt.ylabel('f(b_t)')
plt.legend()

# Plot b_t - a_t vs iterations
plt.subplot(2, 2, 3)
plt.plot(np.arange(0, iterations + 1), diff_list, 'go-', label='b_t - a_t vs iterations')
plt.xlabel('iterations')
plt.ylabel('b_t - a_t')
plt.legend()

# Plot ratio of f(b_t) - f(a_t) over previous difference vs iterations
plt.subplot(2, 2, 4)
plt.plot(np.arange(1, iterations + 1), ratio_list, 'mo-', label='(f(b_t)-f(a_t))/(f(b_prev)-f(a_prev))')
plt.xlabel('iterations')
plt.ylabel('ratio')
plt.legend()

# Show the plot
plt.tight_layout()
# plt.savefig("Q4GoldenSectionSearch.png") # Save the plot
plt.show()
print(f_avalues)
print(f_bvalues)
 
############################################################################################################################################################

# 4.2 Fibonacci Search



# Define the function
def f(x):
    return x * (x - 1) * (x - 3) * (x + 2)

# Function to generate Fibonacci numbers
def fibonacci_numbers(n):
    fibs = [0, 1]
    for i in range(2, n+1):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

# Fibonacci Search
def fibonacci_search(a, b, tol=1e-4):
    fibs = fibonacci_numbers(30)  # Generate enough Fibonacci numbers
    n = len(fibs) - 1
    iteration_count = 0  # Initialize iteration counter
    
    f_avalues = [f(a)]
    f_bvalues = [f(b)]
    diff_list = [b - a]
    
    for i in range(2, n):
        iteration_count += 1  # Increment iteration counter
        ratio = 1 - fibs[n - i + 1] / fibs[n - i + 2]
        x1 = a + ratio * (b - a)
        x2 = b - ratio * (b - a)
        
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1

        # Record values
        f_avalues.append(f(a))
        f_bvalues.append(f(b))
        diff_list.append(b - a)
        
        # Stopping condition
        if (b - a) < tol:
            break
    
    x_star = (a + b) / 2  # Midpoint as the best estimate of x*
    return x_star, iteration_count, f_avalues, f_bvalues, diff_list

# Run Fibonacci Search
result, iterations, f_avalues, f_bvalues, diff_list = fibonacci_search(1, 3)
print(f"Fibonacci Search result: x* = {result}, f(x*) = {f(result)}, iterations = {iterations}")
print(f_avalues)
print(f_bvalues)

# Calculate the ratio list
ratio_list = np.zeros(iterations)
for i in range(2, iterations + 1):
    ratio_list[i - 1] = (f_bvalues[i] - f_avalues[i]) / (f_bvalues[i - 1] - f_avalues[i - 1])


# Plotting
plt.figure(figsize=(10, 8))

# Plot f(a_t) vs iterations
plt.subplot(2, 2, 1)
plt.plot(np.arange(0, iterations + 1), f_avalues[0:iterations + 1], 'ro-', label='f(a_t) vs iterations')
plt.xlabel('iterations')
plt.ylabel('f(a_t)')
plt.legend()

# Plot f(b_t) vs iterations
plt.subplot(2, 2, 2)
plt.plot(np.arange(0, iterations + 1), f_bvalues[0:iterations + 1], 'bo-', label='f(b_t) vs iterations')
plt.xlabel('iterations')
plt.ylabel('f(b_t)')
plt.legend()

# Plot b_t - a_t vs iterations
plt.subplot(2, 2, 3)
plt.plot(np.arange(0, iterations + 1), diff_list, 'go-', label='b_t - a_t vs iterations')
plt.xlabel('iterations')
plt.ylabel('b_t - a_t')
plt.legend()

# Plot ratio of f(b_t) - f(a_t) over previous difference vs iterations
plt.subplot(2, 2, 4)
plt.plot(np.arange(1, iterations), ratio_list[1:], 'mo-', label='(f(b_t)-f(a_t))/(f(b_prev)-f(a_prev))')
plt.xlabel('iterations')
plt.ylabel('ratio')
plt.legend()

# Show the plot
plt.tight_layout()
# plt.savefig("Q4FibonacciSearch.png") # Save the plot
plt.show()


############################################################################################################################################################
# End of the code