import numpy as np
import matplotlib.pyplot as plt

def generate_training_data():
    d = 100 # dimensions of data
    n = 1000 # number of data points
    X = np.random.normal(0,1, size=(n,d))
    w_true = np.random.normal(0,1, size=(d,1))
    y = X.dot(w_true) + np.random.normal(0,0.5,size=(n,1))
    return X, y, w_true

def generate_test_data(w_true):
    d = 100 # dimensions of data
    n = 1000 # number of data points
    X = np.random.normal(0,1, size=(n,d))
    y = X.dot(w_true) + np.random.normal(0,0.5,size=(n,1))
    return X, y

def closed_form_solution(Xtrain, ytrain):
    w_ls = np.dot(np.linalg.inv(Xtrain.T @ Xtrain) @ Xtrain.T, ytrain)
    return w_ls

def obj_func(w, X, y):
    ypred = X @ w
    err = np.sum(np.square(y - ypred))
    return err

def gradient(X, y, w):
    grad = (X.T @ X @ w - X.T @ y) * 2
    # grad = X.T @ X @ w - X.T @ y
    return grad

def grad_desc(N, eta, X, y):
    w = np.zeros((np.shape(X)[1], 1))
    val = [obj_func(w, X, y)]
    for i in range(N):
        grad = gradient(X, y, w)
        w = w - grad * eta
        val.append(obj_func(w, X, y))
    return val

def s_grad_desc(N, eta, X, y):
    n = np.shape(X)[0]
    w = np.zeros((np.shape(X)[1], 1))
    val = [obj_func(w, X, y)]
    while(N > 0):
        index = np.random.randint(n)
        xi = X[index].reshape((np.shape(X)[1], 1))
        yi = y[index]
        grad = 2 * xi * (np.dot(w.T, xi) - yi)
        w = w - grad * eta
        val.append(obj_func(w, X, y))
        N -= 1
    return val

def main():
	#==================Problem Set 1.1=======================

    Xtrain, ytrain, w_true = generate_training_data()
    Xtest, ytest = generate_test_data(w_true)
    w_ls = closed_form_solution(Xtrain, ytrain)
    w_ls_train_err = obj_func(w_ls, Xtrain, ytrain)
    w_0 = np.zeros((len(w_ls),1))
    w_0_train_err = obj_func(w_0, Xtrain, ytrain)
    w_ls_test_err = obj_func(w_ls, Xtest, ytest)
    gap = w_ls_test_err - w_ls_train_err
    print("The total error of w_LS over training set is", w_ls_train_err, "in Problem Set 1.1")
    print()
    print("The total error of w_0 over training set is", w_0_train_err, "in Problem Set 1.1")
    print()
    print("The total error of w_LS over test set is", w_ls_test_err, "in Problem Set 1.1")
    print()
    print("The gap is", gap, "in Problem Set 1.1")
    print()

	#==================Problem Set 1.2=======================

    val1 = grad_desc(20, 0.00005, Xtrain, ytrain)
    val2 = grad_desc(20, 0.0005, Xtrain, ytrain)
    val3 = grad_desc(20, 0.0007, Xtrain, ytrain)
    best_val = min(val1[-1], val2[-1], val3[-1])
    best_step = 0.00005 if val1[-1] == best_val else 0.0005 if val2[-1] == best_val else 0.0007
    print("The step size of", best_step, "has the best final objective function value of", best_val, \
        "in Problem Set 1.2")
    print()
    x_axis = np.linspace(0, 20, 21)
    plt.figure()
    plt.plot(x_axis, val1, label="step size = 0.00005")
    plt.plot(x_axis, val2, label="step size = 0.0005")
    plt.plot(x_axis, val3, label="step size = 0.0007")
    plt.xticks(np.arange(0, 21, 1))
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xlabel("iterations")
    plt.ylabel("objective function value")
    plt.title("Objective function value over iterations(GD)")
    plt.legend()
    plt.savefig("GD.png", bbox_inches="tight")
	
	#==================Problem Set 1.3=======================

    val1s = s_grad_desc(1000, 0.0005, Xtrain, ytrain)
    val2s = s_grad_desc(1000, 0.005, Xtrain, ytrain)
    val3s = s_grad_desc(1000, 0.01, Xtrain, ytrain)
    best_vals = min(val1s[-1], val2s[-1], val3s[-1])
    best_steps = 0.0005 if val1s[-1] == best_vals else 0.005 if val2s[-1] == best_vals else 0.01
    print("The step size of", best_steps, "has the best final objective function value of", best_vals, \
        "in Problem Set 1.3")
    print()
    x_axis = np.linspace(0, 1000, 1001)
    plt.figure()
    plt.plot(x_axis, val1s, label="step size = 0.0005")
    plt.plot(x_axis, val2s, label="step size = 0.005")
    plt.plot(x_axis, val3s, label="step size = 0.01")
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xlabel("iterations")
    plt.ylabel("objective function value")
    plt.title("Objective function value over iterations(SGD)")
    plt.legend()
    plt.savefig("SGD.png", bbox_inches="tight")
 
if __name__ == "__main__":
    main()