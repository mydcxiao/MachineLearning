import matplotlib.pyplot as plt
import numpy as np

def generate_data():
    np.random.seed(42)
    d = 100 # dimensions of data
    n = 1000 # number of data points
    hf_train_sz = int(0.8 * n//2)
    X_pos = np.random.normal(size=(n//2, d))
    X_pos = X_pos + .12
    X_neg = np.random.normal(size=(n//2, d))
    X_neg = X_neg - .12
    X_train = np.concatenate([X_pos[:hf_train_sz],
    X_neg[:hf_train_sz]])
    X_test = np.concatenate([X_pos[hf_train_sz:],
    X_neg[hf_train_sz:]])
    y_train = np.concatenate([np.ones(hf_train_sz),
    -1 * np.ones(hf_train_sz)])
    y_test = np.concatenate([np.ones(n//2 - hf_train_sz),
    -1 * np.ones(n//2 - hf_train_sz)])
    return X_train, y_train.reshape(len(y_train), 1), X_test, y_test.reshape(len(y_test), 1)

def obj_func(w, X, y):
    ypred = X @ w
    err = np.sum(np.log(1 + np.exp(np.multiply(-y, ypred)))) / np.shape(y)[0]
    return err

def zero_one_loss(w, X, y):
    ypred = X @ w
    n = np.shape(y)[0]
    ypred = np.where(ypred > 0, np.ones((n, 1)), -np.ones((n, 1)))
    err = np.count_nonzero(y - ypred) / n
    return err

def sgd(N, eta, X, y, X_test, y_test):
    n = np.shape(X)[0]
    w = np.zeros((np.shape(X)[1], 1))
    val = [obj_func(w, X, y)]
    val1 = [obj_func(w, X_test, y_test)]
    val2 = [zero_one_loss(w, X_test, y_test)]
    while(N > 0):
        index = np.random.randint(n)
        xi = X[index].reshape((np.shape(X)[1], 1))
        yi = y[index]
        grad = - yi / (1 + np.exp(w.T @ xi * yi)) * xi
        w = w - grad * eta
        val.append(obj_func(w, X, y))
        val1.append(obj_func(w, X_test, y_test))
        val2.append(zero_one_loss(w, X_test, y_test))
        N -= 1
    return w, val, val1, val2

def main():
    X_train, y_train, X_test, y_test = generate_data()
    step = [0.0005, 0.005, 0.05]
    
    # Problem 4.1
    _, train_obj_l, test_obj_l, zero_one_l = map(list,zip(*\
        [sgd(5000, i, X_train, y_train, X_test, y_test) for i in step]))
    x_axis = np.linspace(0, 5000, 5001)
    plt.figure()
    plt.plot(x_axis, train_obj_l[0], label="training error, step size = 0.0005")
    plt.plot(x_axis, train_obj_l[1], label="training error, step size = 0.005")
    plt.plot(x_axis, train_obj_l[2], label="training error, step size = 0.05")
    plt.plot(x_axis, test_obj_l[0], label="test error, step size = 0.0005")
    plt.plot(x_axis, test_obj_l[1], label="test error, step size = 0.005")
    plt.plot(x_axis, test_obj_l[2], label="test error, step size = 0.05")
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xlabel("iterations")
    plt.ylabel("objective function value")
    plt.title("Objective Function Value over Iterations")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("P4-1.png", bbox_inches="tight")
    
    #Problem 4.2
    plt.figure()
    plt.plot(x_axis, zero_one_l[0], label="step size = 0.0005")
    plt.plot(x_axis, zero_one_l[1], label="step size = 0.005")
    plt.plot(x_axis, zero_one_l[2], label="step size = 0.05")
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xlabel("iterations")
    plt.ylabel("0-1 loss")
    plt.title("O-1 Loss on Test Data over Iterations")
    plt.legend()
    plt.savefig("P4-2.png", bbox_inches="tight")

    best_zero_one = min(zero_one_l[0][-1], zero_one_l[1][-1], zero_one_l[2][-1])
    best_step_size = 0.0005 if best_zero_one == zero_one_l[0][-1] else 0.005 if best_zero_one == zero_one_l[1][-1]\
        else 0.05
    print("===========================================================================================")
    print("The step size", best_step_size,"has lowest 0-1 loss of", best_zero_one, "in Problem Set 4.2")
    print()
    

if __name__ == '__main__':
    main()