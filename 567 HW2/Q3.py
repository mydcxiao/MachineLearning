
import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    train_n = 100
    test_n = 1000
    d = 100
    X_train = np.random.normal(0,1, size=(train_n,d))
    w_true = np.random.normal(0,1, size=(d,1))
    y_train = X_train.dot(w_true) + np.random.normal(0,0.5,size=(train_n,1))
    X_test = np.random.normal(0,1, size=(test_n,d))
    y_test = X_test.dot(w_true) + np.random.normal(0,0.5,size=(test_n,1))
    return w_true, X_train, y_train, X_test, y_test


def closed_form(X, y):
    return np.linalg.inv(X) @ y

def norm_error(w, X, y):
    return np.linalg.norm(X @ w - y) / np.linalg.norm(y)

def average_error(X_train_l, y_train_l, X_test_l, y_test_l):
    train_total_error = 0
    test_total_error = 0
    for i in range(len(X_train_l)):
        X_train, y_train, X_test, y_test = X_train_l[i], y_train_l[i], X_test_l[i], y_test_l[i]
        w = closed_form(X_train, y_train)
        train_total_error += norm_error(w, X_train, y_train)
        test_total_error += norm_error(w, X_test, y_test)
    train_average_error = train_total_error / len(X_train_l)
    test_average_error = test_total_error / len(X_train_l)
    return train_average_error, test_average_error

def closed_form_l2(X, y, Lambda):
    return np.linalg.inv(X.T @ X + Lambda * np.eye(np.shape(X)[1])) @ X.T @ y

def average_error_l2(X_train_l, y_train_l, X_test_l, y_test_l, Lambda):
    train_total_error = 0
    test_total_error = 0
    for i in range(len(X_train_l)):
        X_train, y_train, X_test, y_test = X_train_l[i], y_train_l[i], X_test_l[i], y_test_l[i]
        w = closed_form_l2(X_train, y_train, Lambda)
        train_total_error += norm_error(w, X_train, y_train)
        test_total_error += norm_error(w, X_test, y_test)
    train_average_error = train_total_error / len(X_train_l)
    test_average_error = test_total_error / len(X_train_l)
    return train_average_error, test_average_error

def obj_func(w, X, y):
    ypred = X @ w
    err = np.sum(np.square(y - ypred))
    return err

def sgd(N, eta, X, y, w):
    n = np.shape(X)[0]
    # w = np.zeros((np.shape(X)[1], 1))
    val = [obj_func(w, X, y)]
    while(N > 0):
        index = np.random.randint(n)
        xi = X[index].reshape((np.shape(X)[1], 1))
        yi = y[index]
        grad = 2 * xi * (np.dot(w.T, xi) - yi)
        w = w - grad * eta
        val.append(obj_func(w, X, y))
        N -= 1
    return w, val

def average_error_sgd(X_train_l, y_train_l, X_test_l, y_test_l, N, eta, w_0):
    train_total_error = 0
    test_total_error = 0
    for i in range(len(X_train_l)):
        X_train, y_train, X_test, y_test = X_train_l[i], y_train_l[i], X_test_l[i], y_test_l[i]
        w = sgd(N, eta, X_train, y_train, w_0)[0]
        train_total_error += norm_error(w, X_train, y_train)
        test_total_error += norm_error(w, X_test, y_test)
    train_average_error = train_total_error / len(X_train_l)
    test_average_error = test_total_error / len(X_train_l)
    return train_average_error, test_average_error

def average_error_true(w_true_l, X_train_l, y_train_l, X_test_l, y_test_l):
    train_total_error = 0
    test_total_error = 0
    for i in range(len(X_train_l)):
        w_true, X_train, y_train, X_test, y_test = w_true_l[i], X_train_l[i], y_train_l[i], X_test_l[i], y_test_l[i]
        train_total_error += norm_error(w_true, X_train, y_train)
        test_total_error += norm_error(w_true, X_test, y_test)
    train_average_error = train_total_error / len(X_train_l)
    test_average_error = test_total_error / len(X_train_l)
    return train_average_error, test_average_error

def sgd_norm(N, eta, X, y, X_test, y_test):
    n = np.shape(X)[0]
    w = np.zeros((np.shape(X)[1], 1))
    train_error = [norm_error(w, X, y)]
    test_error = [norm_error(w, X_test, y_test)]
    w_norm = [np.linalg.norm(w)]
    while(N > 0):
        index = np.random.randint(n)
        xi = X[index].reshape((np.shape(X)[1], 1))
        yi = y[index]
        grad = 2 * xi * (np.dot(w.T, xi) - yi)
        w = w - grad * eta
        train_error.append(norm_error(w, X, y))
        w_norm.append(np.linalg.norm(w))
        if(N % 100 == 1):
            test_error.append(norm_error(w, X_test, y_test))
        N -= 1
    return w, train_error, test_error, w_norm

def rand_init(X, r):
    w_0 = np.random.normal(0,1, size=(np.shape(X)[1],1))
    w_0 = r * w_0 / np.linalg.norm(w_0)
    return w_0

def main():
    w_true_l, X_train_l, y_train_l, X_test_l, y_test_l =  map(list,zip(*[generate_data() for i in range(10)]))
    #Problem 3.1
    train_average_error, test_average_error = average_error(X_train_l, y_train_l, X_test_l, y_test_l)
    print("======================================================================================================")
    print("The averaged normalized training error of closed form is", train_average_error,"in Problem Set 3.1")
    print()
    print("The averaged normalized test error of closed form is", test_average_error,"in Problem Set 3.1")
    print()

    #Problem 3.2
    Lambda = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
    train_average_error_l2, test_average_error_l2 =  map(list,zip(*[average_error_l2(X_train_l, y_train_l, X_test_l, y_test_l, l) for l in Lambda]))
    plt.figure()
    plt.plot(Lambda, train_average_error_l2, label="training error")
    plt.plot(Lambda, test_average_error_l2, label="test error")
    plt.xscale('log')
    plt.xticks(5 * np.logspace(-4, 2, 7), [0.0005, 0.005, 0.05, 0.5, 5, 50, 500])
    plt.minorticks_off()
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xlabel("lambda")
    plt.ylabel("averaged normalized error")
    plt.title("Averaged Normalized Error over Lambda")
    plt.legend()
    plt.savefig("P3-2.png", bbox_inches="tight")
    
    #Problem 3.3
    w_0 = np.zeros((np.shape(X_train_l[0])[1], 1))
    step = [0.00005, 0.0005, 0.005]
    train_average_error_sgd, test_average_error_sgd = map(list,zip(*[\
        average_error_sgd( X_train_l, y_train_l, X_test_l, y_test_l, 1000000, eta, w_0)\
            for eta in step]))
    for i in range(len(step)):
        print("The averaged normalized training error of sgd with step size", step[i],"is", train_average_error_sgd[i],"in Problem Set 3.3")
        print()
        print("The averaged normalized test error of sgd with step size", step[i],"is", test_average_error_sgd[i],"in Problem Set 3.3")
        print()
    train_average_error_true, test_average_error_true = average_error_true(w_true_l, X_train_l, y_train_l, X_test_l, y_test_l)
    print("The averaged normalized training error of w_true is", train_average_error_true,"in Problem Set 3.3")
    print()
    print("The averaged normalized test error of w_true is", test_average_error_true,"in Problem Set 3.3")
    print()

    #Problem 3.4
    w_true, X_train, y_train, X_test, y_test = generate_data()
    ETA = [0.00005, 0.005]
    _, train_error, test_error, w_norm = map(list,zip(*\
        [sgd_norm(1000000, eta, X_train, y_train, X_test, y_test) for eta in ETA]))
    #train error plot
    x_axis = np.linspace(0, 1000000, 1000001)
    y_axis = norm_error(w_true, X_train, y_train) * np.ones((1000001, 1))
    plt.figure()
    plt.plot(x_axis, train_error[0], label="step size = 0.00005")
    plt.plot(x_axis, train_error[1], label="step size = 0.005")
    plt.plot(x_axis, y_axis, label="true model")
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xlabel("iterations")
    plt.ylabel("normalized training error")
    plt.title("Normalized Training Error over Iterations(SGD)")
    plt.legend()
    plt.savefig("P3-4-1.png", bbox_inches="tight")

    #test error plot
    x_axis = np.linspace(0, 1000000, 10001)
    plt.figure()
    plt.plot(x_axis, test_error[0], label="step size = 0.00005")
    plt.plot(x_axis, test_error[1], label="step size = 0.005")
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xlabel("iterations")
    plt.ylabel("normalized test error")
    plt.title("Normalized Test Error over Iterations(SGD)")
    plt.legend()
    plt.savefig("P3-4-2.png", bbox_inches="tight")

    #l2 norm plot
    x_axis = np.linspace(0, 1000000, 1000001)
    plt.figure()
    plt.plot(x_axis, w_norm[0], label="step size = 0.00005")
    plt.plot(x_axis, w_norm[1], label="step size = 0.005")
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xlabel("iterations")
    plt.ylabel("L2 norm of w")
    plt.title("L2 Norm of w over Iterations(SGD)")
    plt.legend()
    plt.savefig("P3-4-3.png", bbox_inches="tight")

    #Problem 3.5
    rad = [0, 0.1, 0.5, 1, 10, 20, 30]
    w_0_l = [rand_init(X_train_l[0], r) for r in rad]
    train_average_error_rand_init, test_average_error_rand_init = map(list,zip(*[\
        average_error_sgd( X_train_l, y_train_l, X_test_l, y_test_l, 1000000, 0.00005, w_0)\
            for w_0 in w_0_l]))
    x_axis = range(len(rad))
    plt.figure()
    plt.plot(x_axis, train_average_error_rand_init, label="training error")
    plt.plot(x_axis, test_average_error_rand_init, label="test error")
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.xticks(x_axis, rad)
    plt.xlabel("radius r")
    plt.ylabel("Averaged normalized error")
    plt.title("Averaged Normalized Error over Radius(SGD)")
    plt.legend()
    plt.savefig("P3-5.png", bbox_inches="tight")

    # write answers into .txt file
    output_file = 'Q3_output.txt'
    f=open(output_file, 'w')
    f.write("The averaged normalized training error of closed form is %e in Problem Set 3.1\n" % (train_average_error))
    f.write("The averaged normalized test error of closed form is %.6f in Problem Set 3.1\n" % (test_average_error))
    f.write("Average error for l2 regularization(lambda, train error, test error):\n")
    for i in range(len(test_average_error_l2)):
        f.write('%f %.6f %.6f' % (Lambda[i], train_average_error_l2[i], test_average_error_l2[i])+'\n')
    f.write("Average error for different step size of SGD(step size, train error, test error):\n")
    for i in range(len(train_average_error_sgd)):
        f.write('%f %.6f %.6f' % (step[i], train_average_error_sgd[i], test_average_error_sgd[i])+'\n')
    f.write("The averaged normalized training error of w_true is %.6f in Problem Set 3.3\n" % (train_average_error_true))
    f.write("The averaged normalized test error of w_true is %.6f in Problem Set 3.3\n" % (test_average_error_true))
    f.write("Average error for different radius(radius, train error, test error):\n")
    for i in range(len(rad)):
        f.write('%f %.6f %.6f' % (rad[i], train_average_error_rand_init[i], test_average_error_rand_init[i])+'\n')
    f.close()

if __name__=="__main__" :
    main()