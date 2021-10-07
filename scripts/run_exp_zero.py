"""
Script to run the experiment  on zero-shot adaptation.
"""
import numpy as np
import argparse
import os

from tqdm import tqdm

import sys
sys.path.append('..')
sys.path.append('/Users/shiro/research/projects/symmetry-bias/')
from utils.visualization import draw_bar_plot
from utils.exp_management import ExperimentManager
from utils.statistical_tests import run_statistical_tests


def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    seeds = np.arange(args.num_seeds)

    accuracies_symmetry = []
    accuracies_vanilla = []

    for seed in tqdm(seeds):

        exp_manager = ExperimentManager(seed=seed, data_dim=10)

        exp_manager.run_train(train_iterations=1000)
        exp_manager.run_test(test_iterations=1)

        accuracies_symmetry.append(
            float(exp_manager.accuracies_backward[0]))
        accuracies_vanilla.append(
            float(exp_manager.accuracies_backward_test_only[0]))

    save_path = save_dir + 'accuracy_step_zero.pdf'

    draw_bar_plot(
        result_1=accuracies_vanilla,
        result_2=accuracies_symmetry,
        labels=['Vanilla', 'Symmetry Bias'],
        title='Zero-shot Performance',
        xlabel='Model',
        ylabel='Accuracy',
        save_path=save_path
    )

    save_path = save_dir + 'stat_test_result.txt'
    run_statistical_tests(
        save_path=save_path,
        result_1=accuracies_vanilla,
        result_2=accuracies_symmetry,
        is_pair=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument(
        '--save_dir', type=str,
        default='/Users/shiro/research/projects/symmetry-bias/results/zero-shot/'
                        )
    args = parser.parse_args()
    main(args)
