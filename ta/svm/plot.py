# Created Date: Sat, Sep 23rd 2023
# Author: F. Waskito
# Last Modified: Sat, Sep 23rd 2023 11:09:48 AM

import numpy
import pylab


def plot_vector(X, y, legend):
    n_samples = X.shape[0]
    classes = list(set(y))
    x_size = 1000 / n_samples
    colors = ['#999', '#fff']
    for i, y_i in enumerate(classes):
        pylab.scatter(
            X[y == y_i, 0],
            X[y == y_i, 1],
            color=colors[i],
            edgecolors='#000',
            marker='o',
            label=y_i,
            s=x_size,
        )

    pylab.xlabel(r"$x_1$")
    pylab.ylabel(r"$x_2$")
    pylab.legend(loc=legend)
    pylab.axis("tight")
    pylab.show()


def plot_margin(X, y, clf, legend):
    # objective function
    def f(x, w, b, c=0):
        # given x, return y such that [x, y] lie on w.x + b = c
        return (-w[0] * x - b + c) / w[1]

    b = clf.b
    w = clf.w
    # Plotting the margin (hyperplane and cannonical hyperplane)
    n_samples = X.shape[0]
    classes = list(set(y))
    # a0 = numpy.min(X)
    # a0 = -1
    # b0 = numpy.max(X)
    a0 = numpy.min(X) - numpy.max(X) / n_samples
    b0 = numpy.max(X) + numpy.max(X) / n_samples
    # a0 = b/w[0]
    # b0 = b/w[1]

    # w.x + b = 0
    a1 = f(a0, w, b)
    b1 = f(b0, w, b)
    pylab.plot(
        [a0, b0],
        [a1, b1],
        "b-",
    )

    # w.x + b = 1; and
    # w.x + b = -1
    for y_i in classes:
        a1 = f(a0, w, b, y_i)
        b1 = f(b0, w, b, y_i)
        pylab.plot(
            [a0, b0],
            [a1, b1],
            "b--",
        )

    x_size = 1000 / n_samples
    # Plotting the data
    # a) if data is support vector
    svs = clf.sv
    pylab.scatter(
        svs[:, 0],
        svs[:, 1],
        facecolors="blue",
        label="Support",
        s=x_size * 2,
    )
    # b) if data isn't support vector
    colors = ['#999', '#fff']
    for i, y_i in enumerate(classes):
        pylab.scatter(
            X[y == y_i, 0],
            X[y == y_i, 1],
            color=colors[i],
            edgecolors='#000',
            marker='o',
            label=y_i,
            s=x_size,
        )

    pylab.xlabel(r"$x_1$")
    pylab.ylabel(r"$x_2$")
    pylab.axhline(y=0, color='#000')
    pylab.axvline(x=0, color='#000')
    pylab.legend(loc=legend)
    pylab.axis("tight")
    pylab.show()

    # pylab.scatter(
    #     X[y == 1, 0],
    #     X[y == 1, 1],
    #     color='#fff',
    #     edgecolors='#000',
    #     marker='o',
    #     label=classes[0],
    #     s=x_size,
    # )
    # pylab.scatter(
    #     X[y == -1, 0],
    #     X[y == -1, 1],
    #     color='#6B696B',
    #     edgecolors="#000",
    #     marker='o',
    #     label=classes[1],
    #     s=x_size,
    # )


def plot_contour(X, y, clf):
    # pylab.plot(X[y == 1, 0], X[y == 1, 1], "ro")
    # pylab.plot(X[y == -1, 0], X[y == -1:, 1], "bo")
    # pylab.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

    classes = list(set(y))
    svs = clf.sv
    pylab.scatter(
        svs[:, 0],
        svs[:, 1],
        facecolors="blue",
        label="Support",
        s=200,
    )
    pylab.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color='#fff',
        edgecolors='#000',
        marker='o',
        label=classes[0],
        s=100,
    )
    pylab.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        color='#6B696B',
        edgecolors="#000",
        marker='o',
        label=classes[1],
        s=100,
    )

    n_sampels = X.shape[0]
    amin = numpy.min(X) - (numpy.max(X) / n_sampels) * 2
    amax = numpy.max(X) + (numpy.max(X) / n_sampels) * 2

    # print(f"> n_samples: {n_sampels}")
    # print(f"> x_max    : {amax}")
    # print(f"> x_min    : {amin}")

    X1, X2 = numpy.meshgrid(
        numpy.linspace(amin, amax, n_sampels),
        numpy.linspace(amin, amax, n_sampels),
    )

    # print(f"\n> X1: {X1}")
    # print(f"> X2: {X2}")
    # print(f"> X : {X}")

    X = numpy.array([[x1, x2] for x1, x2 in zip(
        numpy.ravel(X1),
        numpy.ravel(X2),
    )])

    print(f"\n> X1: {X1}")
    print(f"> X2: {X2}")

    Z = clf.project(X).reshape(X1.shape)
    pylab.contour(
        X1,
        X2,
        Z,
        [0.0],
        colors='k',
        linewidths=1,
        origin="lower",
    )
    pylab.contour(
        X1,
        X2,
        Z + 1,
        [0.0],
        colors='grey',
        linewidths=1,
        origin="lower",
    )
    pylab.contour(
        X1,
        X2,
        Z - 1,
        [0.0],
        colors='grey',
        linewidths=1,
        origin="lower",
    )
    pylab.xlabel(r"$x_1$")
    pylab.ylabel(r"$x_2$")
    pylab.legend(loc="upper right")
    pylab.axis("tight")
    pylab.show()