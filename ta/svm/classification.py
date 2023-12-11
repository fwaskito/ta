# Created Date: Mon, Sep 18th 2023
# Author: F. Waskito
# Last Modified: Thu, Sep 21st 2023 7:51:57 PM

import numpy
import cvxopt
from numpy import array, ndarray
from svm.plot import plot_margin, plot_contour

class Kernel:
    def _linear_kernel(self, x1, x2):
        return numpy.dot(x1, x2)

    def _polynomial_kernel(self, x, y, p):
        return (1 + numpy.dot(x, y))**p

    def _gaussian_kernel(self, x, y, sigma):
        return numpy.exp(-numpy.linalg.norm(x - y)**2 / (2 * (sigma**2)))

    def map_vector(
        self,
        x1,
        x2,
        kernel=None,
        p=None,
        sigma=5.0,
    ):
        if kernel == "linear":
            return self._linear_kernel(x1, x2)
        if kernel == "polynomial":
            if p:
                return self._polynomial_kernel(x1, x2, p=p)
            return self._polynomial_kernel(x1, x2, p=3)
        if kernel == "rbf":
            if sigma:
                return self._gaussian_kernel(x1, x2, sigma=sigma)
            return self._gaussian_kernel(x1, x2, sigma=5.0)

        print(f"The kerne names '{kernel}' not found!")
        return None


class SVClassifier(Kernel):
    def __init__(
        self,
        kernel: str = "linear",
        C: float = None,
        p: int = None,
        sigma: float = None,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.p = p
        self.sigma = sigma
        if self.C: self.C = float(self.C)
        if self.p: self.p = int(p)
        if self.sigma: self.sigma = float(self.sigma)

    def fit(
        self,
        X: ndarray,
        y: array,
    ):
        n_samples, n_features = X.shape

        # Gram matrix
        K = numpy.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.map_vector(
                    X[i],
                    X[j],
                    self.kernel,
                    self.p,
                    self.sigma,
                )

        P = cvxopt.matrix(numpy.outer(y, y) * K)
        q = cvxopt.matrix(numpy.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        A = cvxopt.matrix(numpy.array(A).astype('float'))
        b = cvxopt.matrix(0.0)

        if self.C:
            temp1 = numpy.diag(numpy.ones(n_samples) * -1)
            temp2 = numpy.identity(n_samples)
            G = cvxopt.matrix(numpy.vstack((temp1, temp2)))
            temp1 = numpy.zeros(n_samples)
            temp2 = numpy.ones(n_samples) * self.C
            h = cvxopt.matrix(numpy.hstack((temp1, temp2)))
        else:
            G = cvxopt.matrix(numpy.diag(numpy.ones(n_samples) * -1))
            h = cvxopt.matrix(numpy.zeros(n_samples))

        # Setting options
        cvxopt.solvers.options["show_progress"] = True
        cvxopt.solvers.options["absto1"] = 1e-10
        cvxopt.solvers.options["relto1"] = 1e-10
        cvxopt.solvers.options["feasto1"] = 1e-10

        # Solve quadratic programming (QP) problem
        qp_solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = numpy.ravel(qp_solution['x']) # alpha

        # Support vectors have non-zero lagrange multipliers
        sv_idx = a > 1e-5
        ind = numpy.arange(len(a))[sv_idx]
        self.a = a[sv_idx]
        self.sv = X[sv_idx]
        self.sv_y = y[sv_idx]

        print(f"\n{len(self.a)} support vectors out of {n_samples} points.")
        print(f"> SV        : {self.sv}")
        print(f"> SV labels : {self.sv_y}")

        # Bias (or intercept in linear)
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= numpy.sum(self.a * self.sv_y * K[ind[n], sv_idx])

        self.b /= len(self.a)

        # Weight vector
        if self.kernel == "linear":
            self.w = numpy.zeros(n_features)
            for i, a in enumerate(self.a):
                self.w += a * self.sv_y[i] * self.sv[i]
        else:
            self.w = None

        print("\n------------------")
        print(f"> b: {self.b}")
        print(f"> w: {self.w}")

    def project(self, X):
        if self.w is not None:
            return numpy.dot(X, self.w) + self.b

        y_predict = numpy.zeros(len(X))
        for i, x_i in enumerate(X):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                s += a * sv_y * self.map_vector(
                    x_i,
                    sv,
                    self.kernel,
                    self.p,
                    self.sigma,
                )

            y_predict[i] = s
        return y_predict + self.b

    def predict(self, X):
        return numpy.sign(self.project(X))


def gen_lin_separable_data():
    mean1 = numpy.array([0, 2])
    mean2 = numpy.array([2, 0])
    cov = numpy.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = numpy.random.multivariate_normal(mean1, cov, 100)
    y1 = numpy.ones(len(X1))
    X2 = numpy.random.multivariate_normal(mean2, cov, 100)
    y2 = numpy.ones(len(X2)) * -1
    return X1, y1, X2, y2


def gen_non_lin_separable_data():
    mean1 = numpy.array([-1, 2])
    mean2 = numpy.array([1, -1])
    mean3 = numpy.array([4, -4])
    mean4 = numpy.array([-4, 4])
    cov = numpy.array([[1.0, 0.8], [0.8, 1.0]])
    X1 = numpy.random.multivariate_normal(mean1, cov, 50)
    X1 = numpy.vstack((X1, numpy.random.multivariate_normal(mean3, cov, 50)))
    y1 = numpy.ones(len(X1))
    X2 = numpy.random.multivariate_normal(mean2, cov, 50)
    X2 = numpy.vstack((X1, numpy.random.multivariate_normal(mean4, cov, 50)))
    y2 = numpy.ones(len(X2)) * -1
    return X1, y1, X2, y2


def gen_lin_separable_overlap_data():
    # generate training data in the 2-D case
    mean1 = numpy.array([0, 2])
    mean2 = numpy.array([2, 0])
    cov = numpy.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = numpy.random.multivariate_normal(mean1, cov, 100)
    y1 = numpy.ones(len(X1))
    X2 = numpy.random.multivariate_normal(mean2, cov, 100)
    y2 = numpy.ones(len(X2)) * -1
    return X1, y1, X2, y2


def split_train(X1, y1, X2, y2):
    X1_train = X1[:90]
    y1_train = y1[:90]
    X2_train = X2[:90]
    y2_train = y2[:90]
    X_train = numpy.vstack((X1_train, X2_train))
    y_train = numpy.hstack((y1_train, y2_train))
    return X_train, y_train


def split_test(X1, y1, X2, y2):
    X1_test = X1[90:]
    y1_test = y1[90:]
    X2_test = X2[90:]
    y2_test = y2[90:]
    X_test = numpy.vstack((X1_test, X2_test))
    y_test = numpy.hstack((y1_test, y2_test))
    return X_test, y_test


def test_linear():
    X1, y1, X2, y2 = gen_lin_separable_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    clf = SVClassifier()
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = numpy.sum(y_predict == y_test)
    print(f"{correct} out of {len(y_predict)} predictions correct.")

    plot_margin(X_train, y_train, clf, "upper left")


def test_non_linear():
    X1, y1, X2, y2 = gen_non_lin_separable_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    clf = SVClassifier(kernel="polynomial")
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = numpy.sum(y_predict == y_test)
    print(f"{correct} out of {len(y_predict)} predictions correct.")

    plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)


def test_soft():
    X1, y1, X2, y2 = gen_lin_separable_overlap_data()
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    clf = SVClassifier(C=1000.1)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = numpy.sum(y_predict == y_test)
    print(f"{correct} out of {len(y_predict)} predictions correct.")

    plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)