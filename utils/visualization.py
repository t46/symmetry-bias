"""
Functions for visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5})

colors = [
          '#528B8B',  # Slate Gray 4
          '#FA8072'  # Salmon
          ]


def draw_comparison_line_plot(
                              result_1: List[List],
                              result_2: List[List],
                              labels: List[str],
                              title: str,
                              xlabel: str,
                              ylabel: str,
                              save_path: str
                              ) -> None:
    """Draw line plot for comparison

    Args:
        result_1 (List[List]):
            Result to be compared.
            The length of the outer list is the number of random seeds.
            The inner list is the result of each random seed.
              Ex. Test losses
        result_2 (List[List]):
            Result to be compared.
            The length of the outer list is the number of random seeds.
            The inner list is the result of each random seed.
              Ex. Test losses
        labels (List[str]): Label names for two lines, respectively.
        title (str): Title of the plot.
        xlabel (str): The name of x-axis of the plot.
        ylabel (str): The name of y-axis of the plot.
        save_path (str): The name of the path to save the plot.
    """
    result_1_mean = np.mean(np.array(result_1), axis=0)
    result_1_std = np.std(np.array(result_1), axis=0)
    result_2_mean = np.mean(np.array(result_2), axis=0)
    result_2_std = np.std(np.array(result_2), axis=0)

    iterations = np.arange(result_1_mean.shape[0])

    sns.lineplot(x=iterations, y=result_1_mean,
                 color=colors[0], label=labels[0])
    plt.fill_between(iterations,
                     result_1_mean-result_1_std,
                     result_1_mean+result_1_std,
                     alpha=0.5, color=colors[0]
                     )

    sns.lineplot(x=iterations, y=result_2_mean,
                 color=colors[1], label=labels[1])
    plt.fill_between(iterations,
                     result_2_mean-result_2_std,
                     result_2_mean+result_2_std,
                     alpha=0.5, color=colors[1]
                     )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def draw_box_plot(
                  result_1: List[float],
                  result_2: List[float],
                  labels: List[str],
                  title: str,
                  xlabel: str,
                  ylabel: str,
                  save_path: str
                  ) -> None:
    """Draw box plot for comparison

    Args:
        result_1 (List[float]):
            Result to be compared.
            The length of the outer list is the number of random seeds.
            The inner list is the result of each random seed.
              Ex. The number of steps for convergence
        result_2 (List[float]):
            Result to be compared.
            The length of the outer list is the number of random seeds.
            The inner list is the result of each random seed.
              Ex. The number of steps for convergence
        labels (List[str]): Label names for two lines, respectively.
        title (str): Title of the plot.
        xlabel (str): The name of x-axis of the plot.
        ylabel (str): The name of y-axis of the plot.
        save_path (str): The name of the path to save the plot.
    """
    result = [np.array(result_1), np.array(result_2)]

    sns.boxplot(data=result, palette=colors)
    plt.xticks([0, 1], labels)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def draw_bar_plot(
                  result_1: List[float],
                  result_2: List[float],
                  labels: List[str],
                  title: str,
                  xlabel: str,
                  ylabel: str,
                  save_path: str
                  ) -> None:
    """Draw bar plot for comparison

    Args:
        result_1 (List[float]):
            Result to be compared.
            The length of the outer list is the number of random seeds.
            The inner list is the result of each random seed.
              Ex. The number of steps for convergence
        result_2 (List[float]):
            Result to be compared.
            The length of the outer list is the number of random seeds.
            The inner list is the result of each random seed.
              Ex. The number of steps for convergence
        labels (List[str]): Label names for two lines, respectively.
        title (str): Title of the plot.
        xlabel (str): The name of x-axis of the plot.
        ylabel (str): The name of y-axis of the plot.
        save_path (str): The name of the path to save the plot.
    """
    result = [np.array(result_1), np.array(result_2)]

    sns.barplot(data=result, palette=colors)
    plt.xticks([0, 1], labels)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
