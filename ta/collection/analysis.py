# Created Date: Thu, May 18th 2023
# Author: F. Waskito
# Last Modified: Sun, Jan 21st 2024 11:56:22 PM

import numpy
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter


def get_shape(arr) -> None:
    arr = numpy.array(arr)
    print(f"Shape: {arr.shape}")


def get_distribution(y) -> None:
    counter = Counter(list(y))
    print("Distribution:")
    for item in counter.items():
        print(f"\t{item}")


def plot_class(X, y):
    y = list(y)
    class_distribution = Counter(y)
    for label, _ in class_distribution.items():
        row_ix = numpy.where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))

    plt.title("Samples by Class")
    plt.legend()
    plt.show()


def plot_vector(X, y, legend):
    classes = list(set(y))
    plt.scatter(
        X[y == classes[0], 0],
        X[y == classes[0], 1],
        color="#A50000",
        marker="o",
        label=classes[0],
        s=70,
    )
    plt.scatter(
        X[y == classes[1], 0],
        X[y == classes[1], 1],
        color="#2400A5",
        marker="o",
        label=classes[1],
        s=70,
    )
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend(loc=legend)
    plt.show()
    plt.show()


def plot_decision_boundary(svm_clf, X):
    w_ = svm_clf.coef_[0]
    b_ = svm_clf.intercept_[0]

    x_min = numpy.min(X)
    x_max = numpy.max(X)
    x_points = numpy.linspace(x_min, x_max, 100)
    # At decison boundary
    # w1*x1 + w2*x2 + b = 0
    # => x1 = -w1/w2 * x1 - b/w2
    y_points = -(w_[0] / w_[1]) * x_points - b_ / w_[1]

    margin = 1 / w_[1]
    gutter_up = y_points + margin
    gutter_down = y_points - margin
    svs = svm_clf.support_vectors_

    plt.scatter(
        svs[:, 0],
        svs[:, 1],
        facecolors="#BBB9BB",
        s=250,
    )
    plt.plot(
        x_points,
        y_points,
        "k-",
        linewidth=2,
        label="Hyperplane",
    )
    plt.plot(x_points, gutter_up, "k--", linewidth=1)
    plt.plot(x_points, gutter_down, "k--", linewidth=1)


def plot_hyperplane(svm_clf, data):
    # ---w is the vector of weights---
    w = svm_clf.coef_[0]

    # ---find the slope of the hyperplane---
    slope = -w[0] / w[1]

    # ---bias---
    b = svm_clf.intercept_[0]

    # ---find the coordinate for the hyperplane---
    # x_min = numpy.min(X)
    # x_max = numpy.max(X)
    x_min = 0
    x_max = 1
    xx = numpy.linspace(x_min, x_max)
    yy = slope * xx - (b / w[1])

    # ---plot the margins---
    sv = svm_clf.support_vectors_[0]  # first support vector
    yy_down = slope * xx + (sv[1] - slope * sv[0])
    sv = svm_clf.support_vectors_[-1]  # last support vector
    yy_up = slope * xx + (sv[1] - slope * sv[0])

    # ---plot the points---
    sns.lmplot(
        x="x1",
        y="x2",
        data=data,
        hue="Sentiment",
        palette="Set1",
        fit_reg=False,
        scatter_kws={"s": 70},
    )

    # ---plot the hyperplane---
    plt.plot(xx, yy, linewidth=2, color="green")

    # ---plot the 2 margins---
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy_up, "k--")


def plot_performance_2d(
    res_dict: dict[dict],
    legend: str,
    title: str,
    best_label: bool = True,
    scatter_colors: list[str] = ["red", "blue", "purple"],
    plot_colors: list[str] = ["lightcoral", "cornflowerblue", "orchid"],
):
    axes_labels = []
    for i, (res_key, ax_dict) in enumerate(res_dict.items()):
        if best_label:
            idxmax = ax_dict["Akurasi"].idxmax()

        best = []
        axes = []
        for ax_label, ax_values in ax_dict.items():
            axes.append(ax_values)
            axes_labels.append(ax_label)
            if best_label:
                best.append(ax_values[idxmax])

        plt.scatter(
            axes[0],
            axes[1],
            marker=".",
            color=scatter_colors[i],
            label=res_key,
        )

        plt.plot(
            axes[0],
            axes[1],
            color=plot_colors[i],
        )

        if best_label:
            best_label = f"best {res_key} ({axes_labels[0]}={best[0]})"
            plt.text(
                best[0] * (1 + 0.001),
                best[1] * (1 + 0.001),
                best_label,
                fontsize=8,
            )

    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    plt.legend(loc=legend)
    plt.title(title)
    plt.show()