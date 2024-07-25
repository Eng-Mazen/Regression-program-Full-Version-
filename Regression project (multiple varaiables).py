import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
print("The program accepts only excel sheets files")
path = input("Enter the path of the file using (Ctrl+Shift+C) on the file:\n").strip()
Regression_data = pd.read_excel(fr"{path[1:len(path)-1]}")
test = len(Regression_data.columns)
X = Regression_data.iloc[:, 0:test-1]
X.insert(0, 'X0', 1)
X1 = X.iloc[:, 1]
x  = np.array(X)
y = Regression_data.iloc[:, test-1:test]
y = np.array(y)
theta = np.zeros(len(X.columns)).reshape(len(X.columns), 1)
def caculatecost(X,y,theta):
    m = len(y)
    formula = np.sum((np.dot(X, theta)) - y)
    result = ((formula)**2) / (2 * m)
    return result
print("initial cost =", caculatecost(x, y, theta))
def gradient_descent(iters, alpha, X, y, theta):
    m = len(y)
    history = np.zeros(iters)
    for i in range (iters):
        formula = ((np.dot(X, theta)) - y)
        theta -= X.T.dot(formula) * (alpha / m)
        history[i] = caculatecost(X, y, theta)
    return theta, history
iterations = int(input("Enter the number of iterations: "))
alpha = float(input("Enter the value of alpha: "))
theta, c_new = gradient_descent(iterations, alpha, x, y, theta)
print("New cost =", c_new[iterations-1])
if test in [3,2]:
        if test == 2:
            def darwing_func1(x):
                return theta[0, 0]+theta[1,0] * x
            x_axis_name = input("Enter the x-axis name: ").strip().capitalize()
            y_axis_name = input("Enter the y-axis name: ").strip().capitalize()
            x_start = np.min(x)
            x_end = np.max(x)
            x1 = np.linspace(x_start, x_end, 10000)
            plt.scatter(X1, y, color='b', label="Real values")
            plt.plot(x1, darwing_func1(x1), color='r', label="Regression line")
            plt.xlabel(x_axis_name)
            plt.ylabel(y_axis_name)
            plt.title(f"{x_axis_name} VS {y_axis_name}")
            plt.legend()
            plt.show()
            plt.plot(range(iterations),c_new, color='r')
            plt.xlabel("Iterations")
            plt.ylabel('Cost')
            plt.title("Iterations VS Cost")
            plt.show()
            while True:
                 o = float(input("what is the value you want to predict?:\n"))
                 print(darwing_func1(o))
                 u = input("You want to predict again?:\n").strip().capitalize()
                 if u == "Yes":
                      continue
                 elif u == "No":
                      break
        else:
             axes = plt.figure()
             axes =axes.add_subplot(111, projection='3d')
             x1, x2 = x[:, 1], x[:, 2]
             def drawing_func2(x1,x2):
                  x_height = np.zeros(x1.size).reshape(x1.size, 1)
                  y_height = np.zeros(x2.size).reshape(x2.size, 1)
                  for i in range(x1.size):
                       x_height[i] = theta[0, 0] + theta[1, 0]*x1[i] +theta[2, 0]* x2[i]
                       y_height[i] = theta[0, 0] + theta[1, 0]*x1[i] +theta[2, 0]* x2[i]
                  z_new = np.array([[x_height], [y_height]]).reshape(x_height.size, 2)
                  return z_new
             x_axis_name = input("Enter the x-axis name:\n")
             y_axis_name = input("Enter the y-axis name:\n")
             z_axis_name = input("Enter the z-axis name:\n")
             axes.scatter(x1, x2, y, color='b', label="Real data", alpha=0.4)
             axes.plot_surface(x1.reshape(x1.size, 1), x2.reshape(x2.size, 1), drawing_func2(x1, x2), color="r", label="Regression line")
             axes.set_xlabel(x_axis_name)
             axes.set_ylabel(y_axis_name)
             axes.set_zlabel(z_axis_name)
             axes.legend()
             plt.show()
             plt.plot(range(iterations),c_new, color='r')
             plt.xlabel("Iterations")
             plt.ylabel('Cost')
             plt.title("Iterations VS Cost")
             plt.show()
             while True:
                j = np.zeros(test)
                j[0] = 1
                k = np.zeros(test)
                print("what is the value you want to predict?:\n")
                for i in range(test-1):
                    j[i+1] = float(input(f"Enter the {i+1} value: "))
                for i in range(test):
                    k[i] = theta[i, 0] * j[i]
                print(np.sum(k))
                u = input("You want to predict again?:\n").strip().capitalize()
                if u == "Yes":
                    continue
                elif u == "No":
                    break                                
else:
    print("Sorry this program will return only the cost as its impossible to draw more than 3 dimensions in one graph or less than 2 dimensions")
    while True:
        j = np.zeros(test)
        j[0] = 1
        k = np.zeros(test)
        print("what is the value you want to predict?:\n")
        for i in range(test-1):
             j[i+1] = float(input(f"Enter the {i+1} value: "))
        for i in range(test):
             k[i] = theta[i, 0] * j[i]
        print(np.sum(k))
        u = input("You want to predict again?:\n").strip().capitalize()
        if u == "Yes":
              continue
        elif u == "No":
              break