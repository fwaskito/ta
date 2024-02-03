# Created Date: Thu, May 18th 2023
# Author: F. Waskito
# Last Modified: Fri, Jan 26th 2024 2:25:15 PM

from typing import Union
import numpy
import pandas
import seaborn
from numpy import ndarray
from pandas import DataFrame
from matplotlib import pyplot
from mlxtend.plotting import plot_confusion_matrix
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
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))

    pyplot.title("Samples by Class")
    pyplot.legend()
    pyplot.show()


def plot_vector(X, y, legend):
    classes = list(set(y))
    pyplot.scatter(
        X[y == classes[0], 0],
        X[y == classes[0], 1],
        color="#A50000",
        marker="o",
        label=classes[0],
        s=70,
    )
    pyplot.scatter(
        X[y == classes[1], 0],
        X[y == classes[1], 1],
        color="#2400A5",
        marker="o",
        label=classes[1],
        s=70,
    )
    pyplot.xlabel(r"$x_1$")
    pyplot.ylabel(r"$x_2$")
    pyplot.legend(loc=legend)
    pyplot.show()


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

    pyplot.scatter(
        svs[:, 0],
        svs[:, 1],
        facecolors="#BBB9BB",
        s=250,
    )
    pyplot.plot(
        x_points,
        y_points,
        "k-",
        linewidth=2,
        label="Hyperplane",
    )
    pyplot.plot(x_points, gutter_up, "k--", linewidth=1)
    pyplot.plot(x_points, gutter_down, "k--", linewidth=1)


def plot_hyperplane(svm_clf, data):
    # ---w is the vector of weights---
    w = svm_clf.coef_[0]

    # ---find the slopw of the hyperplane---
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
    seaborn.lmplot(
        x="x1",
        y="x2",
        data=data,
        hue="Sentiment",
        palette="Set1",
        fit_reg=False,
        scatter_kws={"s": 70},
    )

    # ---plot the hyperplane---
    pyplot.plot(xx, yy, linewidth=2, color="green")

    # ---plot the 2 margins---
    pyplot.plot(xx, yy_down, "k--")
    pyplot.plot(xx, yy_up, "k--")


def plot_performance_2d(
    res_dict: dict[dict],
    legend: str,
    title: str = None,
    best_label: bool = True,
    colors: dict[list[str]] = None,
    linestyles: list = None,
    figsize: set = (6.4, 4.8),
    ticksize: list = [10, 10],
):
    ax_aliases = {
        "Iterasi": "i",
        "Akurasi": "acc",
        "C": "$C$",
        "Gamma": "$\gamma$",
        "Degree": "$d$",
    }

    if colors is None:
        colors = {
            "scatter": ["red", "blue", "purple"],
            "plot": ["lightcoral", "cornflowerblue", "orchid"],
        }

    if linestyles is None:
        linestyles = ["-", "--"]

    pyplot.figure(figsize=figsize)
    res_keys = []
    ax_labels = []
    l = 0
    c = 0
    for i, (res_key, ax_dict) in enumerate(res_dict.items()):
        res_keys.append(res_key)
        if best_label:
            idxmax = ax_dict["Akurasi"].idxmax()

        best = []
        axes = []
        for ax_label, ax_values in ax_dict.items():
            axes.append(ax_values)
            ax_labels.append(ax_label)
            if best_label:
                best.append(ax_values[idxmax])

        pyplot.scatter(
            axes[0],
            axes[1],
            marker=".",
            color=colors["scatter"][c],
            label=res_key,
        )

        pyplot.plot(
            axes[0],
            axes[1],
            linestyle=linestyles[l],
            color=colors["plot"][c],
        )

        if best_label:
            best_label = f"best {res_key} ("
            for j, ax_label in enumerate(ax_labels):
                best_label += f"{ax_aliases[ax_label]}={best[j]}"
                if j < len(ax_labels) - 1:
                    best_label += ", "
            best_label += ")"

            pyplot.text(
                best[0] * (1 + 0.001),
                best[1] * (1 + 0.001),
                best_label,
                fontsize=8,
            )

        if i < len(res_dict) - 1:
            ax_labels.clear()

        if c == len(colors["scatter"]) - 1:
            l += 1
            c = 0
        else:
            c += 1

    if "Iterasi" in ax_labels:
        pyplot.xticks(res_dict[res_keys[0]]["Iterasi"])

    if "Degree" in ax_labels:
        pyplot.yticks([3, 6, 9])

    if "C" in ax_labels:
        idx = ax_labels.index("C")
        if idx == 0:
            pyplot.xticks([1, 100, 1000, 10000])
        else:
            pyplot.yticks([1, 100, 1000, 10000])

    if "Gamma" in ax_labels:
        idx = ax_labels.index("Gamma")
        if idx == 0:
            pyplot.xticks([0.01, 0.1, 1.0])
        else:
            pyplot.yticks([0.01, 0.1, 1.0])

    if title:
        pyplot.title(title, fontsize=11)

    pyplot.xticks(fontsize=ticksize[0])
    pyplot.yticks(fontsize=ticksize[1])
    pyplot.xlabel(ax_labels[0])
    pyplot.ylabel(ax_labels[1])
    pyplot.legend(loc=legend)
    pyplot.grid(True)
    pyplot.show()


def plot_performance_3d(
    res_dict: dict[dict],
    legend: str,
    title: str = None,
    best_label: bool = True,
    colors: dict[list[str]] = None,
    figsize: set = (10, 7),
):
    ax_aliases = {
        "Iterasi": "i",
        "Akurasi": "acc",
        "C": "$C$",
        "Gamma": "$\gamma$",
        "Degree": "$d$",
    }
    if colors is None:
        colors = {
            "scatter": ["red", "blue", "purple"],
            "plot": ["lightcoral", "cornflowerblue", "orchid"],
        }

    res_keys = []
    ax_labels = []
    pyplot.figure(figsize=figsize)
    ax = pyplot.axes(projection="3d")
    k = 0
    for i, (res_key, ax_dict) in enumerate(res_dict.items()):
        res_keys.append(res_key)
        if best_label:
            idxmax = ax_dict["Akurasi"].idxmax()

        best = []
        axes = []
        for ax_label, ax_values in ax_dict.items():
            axes.append(ax_values)
            ax_labels.append(ax_label)
            if best_label:
                best.append(ax_values[idxmax])

        ax.scatter3D(
            axes[0],
            axes[1],
            axes[2],
            marker=".",
            color=colors["scatter"][i - k],
            label=res_key,
        )

        ax.plot3D(
            axes[0],
            axes[1],
            axes[2],
            color=colors["plot"][i - k],
        )

        if best_label:
            best_label = f"best {res_key} ("
            for j, ax_label in enumerate(ax_labels):
                best_label += f"{ax_aliases[ax_label]}={best[j]}"
                if j < len(ax_labels) - 1:
                    best_label += ", "
            best_label += ")"

            ax.text(
                best[0] * (1 + 0.001),
                best[1] * (1 + 0.001),
                best[2] * (1 + 0.001),
                best_label,
                fontsize=8,
            )

        if i < len(res_dict) - 1:
            ax_labels.clear()

        if i == len(colors["scatter"]):
            k += len(colors["scatter"])

    if "Degree" in ax_labels:
        ax.set_yticks([3, 6, 9])

    if "C" in ax_labels:
        idx = ax_labels.index("C")
        if idx == 0:
            ax.set_xticks([1, 100, 1000, 10000])
        else:
            ax.set_yticks([1, 100, 1000, 10000])

    if "Gamma" in ax_labels:
        idx = ax_labels.index("Gamma")
        if idx == 0:
            ax.set_xticks([0.01, 0.1, 1.0])
        else:
            ax.set_yticks([0.01, 0.1, 1.0])

    if title:
        pyplot.title(title, fontsize=11)

    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_zlabel(ax_labels[2])
    pyplot.legend(loc=legend)
    pyplot.show()


def plot_heatmap(
    cm: Union[ndarray, DataFrame],
    title: str = None,
    cmap: str = "magma",
) -> None:
    classes = ["negatif", "netral", "positif"]
    if cm is DataFrame:
        cm_table = cm
    else:
        cm_table = pandas.DataFrame(cm)

    cm_table.columns = classes
    cm_table.index = classes
    cm_table.columns.name = "kelas prediksi"
    cm_table.index.name = "kelas aktual"
    seaborn.set(font_scale=1.2)
    seaborn.heatmap(
        cm_table,
        annot=True,
        annot_kws={"size": 14},
        fmt="d",
        cmap=cmap,
        linewidths=1,
        linecolor="k",
    )

    if title:
        pyplot.title(title)

    pyplot.show()


def plot_confmatrix(
    cm: Union[ndarray, DataFrame],
    title: str = None,
    cmap: str = "magma",
    figsize: set = (6.4, 4.8),
) -> None:
    if cm is ndarray:
        cm_array = cm
    else:
        cm_array = cm.to_numpy()

    classes = ["negatif", "netral", "positif"]
    fig, ax = plot_confusion_matrix(
        conf_mat=cm_array,
        colorbar=True,
        class_names=classes,
        cmap=cmap,
        figsize=figsize,
    )

    ax.set_xlabel("kelas prediksi")
    ax.set_ylabel("kelas aktual")
    pyplot.title(title, fontsize=11)
    pyplot.show()
