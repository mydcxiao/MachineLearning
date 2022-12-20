# Code modified from https://scikit-learn.org/1.0/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
# Original Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

########################################
# Define the classifiers to be studied #
########################################
# names = [
#     "Linear SVM (C = 0.025)",
#     "Linear SVM (C = 0.25)",
#     "RBF SVM (C=0.25)",
#     "RBF SVM (C=1.0)",
#     "Logistic Regression\n(reg_lambda = 0.0)",
#     "Logistic Regression\n(reg_lambda = 0.5)"
# ]

# classifiers = [
#     SVC(kernel="linear", C=0.025),  # C controls the penalty term for misclassifying training data
#     SVC(kernel="linear", C=0.25),   # Smaller C means stronger regularization
#     SVC(kernel="rbf", gamma=2, C=0.25),
#     SVC(kernel="rbf", gamma=2, C=1),
#     LogisticRegression(penalty='none'),     # none sets regularization term to 0
#     LogisticRegression(penalty='l2', C=2.0) # C is the inverse regularization strength i.e. smaller C means stronger regularization
# ]

#5.2
# names = [
#     "Linear SVM (C = 0.00025)",
#     "Linear SVM (C = 0.0025)",
#     "Linear SVM (C = 0.025)",
#     "Linear SVM (C = 0.25)",
#     "Linear SVM (C = 2.5)",
#     "Linear SVM (C = 25)",
# ]

# classifiers = [
#     SVC(kernel="linear", C=0.00025),  # C controls the penalty term for misclassifying training data
#     SVC(kernel="linear", C=0.0025),   # Smaller C means stronger regularization
#     SVC(kernel="linear", C=0.025),  # C controls the penalty term for misclassifying training data
#     SVC(kernel="linear", C=0.25),   # Smaller C means stronger regularization
#     SVC(kernel="linear", C=2.5),  # C controls the penalty term for misclassifying training data
#     SVC(kernel="linear", C=25),   # Smaller C means stronger regularizatione
# ]

#5.3
# names = [
#     "RBF SVM (C=0.0025)",
#     "RBF SVM (C=0.025)",
#     "RBF SVM (C=0.25)",
#     "RBF SVM (C=1.0)",
#     "RBF SVM (C=2.5)",
#     "RBF SVM (C=25)",
# ]

# classifiers = [
#     SVC(kernel="rbf", gamma=2, C=0.0025),
#     SVC(kernel="rbf", gamma=2, C=0.025), # Smaller C means stronger regularization
#     SVC(kernel="rbf", gamma=2, C=0.25),
#     SVC(kernel="rbf", gamma=2, C=1.0),
#     SVC(kernel="rbf", gamma=2, C=2.5),
#     SVC(kernel="rbf", gamma=2, C=25),# C is the inverse regularization strength i.e. smaller C means stronger regularization
# ]

# 5.4 
names = [
    "Logistic Regression\n(reg_lambda = 0.0)",
    "Logistic Regression\n(reg_lambda = 0.005)",
    "Logistic Regression\n(reg_lambda = 0.05)",
    "Logistic Regression\n(reg_lambda = 0.5)",
    "Logistic Regression\n(reg_lambda = 5)",
    "Logistic Regression\n(reg_lambda = 50)"
]

classifiers = [
    LogisticRegression(penalty='none'),     # none sets regularization term to 0
    LogisticRegression(penalty='l2', C=200.0),   # Smaller C means stronger regularization
    LogisticRegression(penalty='l2', C=20.0),     # none sets regularization term to 0
    LogisticRegression(penalty='l2', C=2.0), 
    LogisticRegression(penalty='l2', C=0.2),     # none sets regularization term to 0
    LogisticRegression(penalty='l2', C=0.02) # C is the inverse regularization strength i.e. smaller C means stronger regularization
]

######################################
# Sample the classification datasets #
######################################
# We will look at two datasets where the classes are not linearly separable.
# The third dataset is linearly separable save for a few noisy datapoints.
X, y = make_classification(
    n_samples=50, n_features=2, n_redundant=0, n_informative=2, random_state=1, 
    n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    ('MOON', make_moons(noise=0.3, random_state=0)),
    ('CIRCLE', make_circles(noise=0.2, factor=0.5, random_state=1)),
    ('LINEARLY_SEPARABLE', linearly_separable),
]


# datasets = [
#     ('LINEARLY_SEPARABLE', linearly_separable),
#     ('MOON', make_moons(noise=0.3, random_state=0)),
#     ('CIRCLE', make_circles(noise=0.2, factor=0.5, random_state=1)),
# ]

#5.2
# datasets = [
#     ('LINEARLY_SEPARABLE', linearly_separable),
#     ('LINEARLY_SEPARABLE', linearly_separable),
#     ('LINEARLY_SEPARABLE', linearly_separable)
# ]

############################
# Plot classifier behavior #
############################
h = 0.02  # step size in the mesh
figure = plt.figure(constrained_layout=True, figsize=(15, 15))
subfigs = figure.subfigures(len(datasets), 1)
subfig_arr = [subfig for subfig in subfigs.flat]

# iterate over datasets
for ds_cnt, (ds_nm, ds) in enumerate(datasets):
    subfig = subfig_arr[ds_cnt]
    subfig.suptitle(ds_nm, fontsize='x-large')
    
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Col 1: just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    axs = subfig.subplots(2, len(classifiers) + 1)
    i = 0
    ax_tr = axs[0, i]
    ax_te = axs[1, i]
    if ds_cnt == 0:
        ax_tr.set_title("Input data")
    # Plot the training points
    ax_tr.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", marker='^')
    # Plot the testing points
    ax_te.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax_tr.set_xlim(xx.min(), xx.max())
    ax_tr.set_ylim(yy.min(), yy.max())
    ax_tr.set_xticks(())
    ax_tr.set_yticks(())
    ax_tr.text(
        xx.max() - 0.3,
        yy.min() + 0.3,
        "Train",
        size=15,
        horizontalalignment="right",
    )
    
    ax_te.set_xlim(xx.min(), xx.max())
    ax_te.set_ylim(yy.min(), yy.max())
    ax_te.set_xticks(())
    ax_te.set_yticks(())
    
    ax_te.text(
        xx.max() - 0.3,
        yy.min() + 0.3,
        "Test",
        size=15,
        horizontalalignment="right",
    )
    
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax_tr = axs[0, i]
        ax_te = axs[1, i]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        score_train = clf.score(X_train, y_train)
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax_tr.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
        ax_te.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot the training points
        ax_tr.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", marker='^'
        )
        # Plot the testing points
        ax_te.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax_tr.set_xlim(xx.min(), xx.max())
        ax_tr.set_ylim(yy.min(), yy.max())
        ax_tr.set_xticks(())
        ax_tr.set_yticks(())
        ax_tr.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score_train).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        if ds_cnt == 0:
            # ax_tr.set_title(name, {'fontsize': 10})
            ax_tr.set_title(name)
        ax_te.set_xlim(xx.min(), xx.max())
        ax_te.set_ylim(yy.min(), yy.max())
        ax_te.set_xticks(())
        ax_te.set_yticks(())
        ax_te.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1
# plt.savefig("P5.png", bbox_inches='tight')
plt.show()