import matplotlib.pyplot as plt
import seaborn as sns
import os

directory = os.getcwd() + "/plots"

if not os.path.exists(directory):
    os.makedirs(directory)


def add_standard_layout_to_plot():
    sns.despine()
    sns.set(style="whitegrid", font_scale=1.5)


def create_bar_plot(df, name="bar_plot", x_label=None, y_label=None, title=None, bar_width=0.4, legend=False, save=False):
    """
    Saves a bar plot with the given options to the /plots directory

    Parameters
    ----------
    df: Pandas dataframe or series
    name: String
    x_label: String
    y_label: String
    bar_width: Float
    legend: Boolean     Whether to show the legend
    save: Boolean       Whether to save the plot
    """

    add_standard_layout_to_plot()
    df.plot(kind="bar", width=bar_width)

    if legend:
        plt.legend()

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    if title:
        plt.title(title)

    plt.xticks(rotation=0)

    plt.show()

    if save:
        plt.savefig(directory + "/" + name)


def create_scatter_plot(df, name="scatter_plot", x_label=None, y_label=None, legend=False):
    """
    Saves a scatter plot with the given options to the /plots directory

    Parameters
    ----------
    df: Pandas dataframe,
    name: String,
    x_label: String,
    y_label: String,
    legend: Boolean
    """

    df.plot(kind="scatter")

    if legend:
        plt.legend()

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    plt.savefig(directory + "/" + name)


def create_hist_plot(df, name="histogram", x_label=None, y_label=None, hist_width=0.1, legend=False):
    """
    Saves a histogram plot with the given options to the /plots directory

    Parameters
    ----------
    df: Pandas dataframe or series
    name: String
    x_label: String
    y_label: String
    hist_width: Float
    legend: Boolean
    """

    df.plot(kind="hist", width=hist_width)

    if legend:
        plt.legend()

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    plt.savefig(directory + "/" + name)


def create_boxplot(df, name="box_plot", x_label=None, y_label=None, legend=False):
    """
    Saves a boxplot with the given options to the /plots directory

    Parameters
    ----------
    df: Pandas dataframe,
    name: String,
    x_label: String,
    y_label: String,
    legend: Boolean
    """

    df.plot(kind="box")

    if legend:
        plt.legend()

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    plt.savefig(directory + "/" + name)
