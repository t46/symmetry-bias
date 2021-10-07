"""
Script to run the experiment to compare the number of steps to converge
"""
import numpy as np
import argparse
import os

from tqdm import tqdm

import sys
sys.path.append('..')
from utils.visualization import draw_comparison_line_plot, draw_box_plot
from utils.exp_management import ExperimentManager
from utils.statistical_tests import run_statistical_tests


def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    seeds = np.arange(args.num_seeds)

    test_losses_symmetry = []
    test_losses_vanilla = []

    steps_to_converge_symmetry = []
    steps_to_converge_vanilla = []

    for seed in tqdm(seeds):

        exp_manager = ExperimentManager(seed=seed, data_dim=10)

        exp_manager.run_train(train_iterations=1000)
        exp_manager.run_test(test_iterations=500)

        test_losses_symmetry.append(exp_manager.test_losses_backward)
        test_losses_vanilla.append(exp_manager.test_losses_backward_test_only)

        steps_to_converge_symmetry.append(
            exp_manager.steps_to_converge)
        steps_to_converge_vanilla.append(
            exp_manager.steps_to_converge_test_only)

    save_path = save_dir + 'test_loss_comparison.pdf'

    draw_comparison_line_plot(
        result_1=test_losses_vanilla,
        result_2=test_losses_symmetry,
        labels=['Vanilla', 'Symmetry Bias'],
        title='Test Loss',
        xlabel='Iteration',
        ylabel='Test Loss',
        save_path=save_path
    )

    save_path = save_dir + 'steps_to_converge_comparison.pdf'

    draw_box_plot(
        result_1=steps_to_converge_vanilla,
        result_2=steps_to_converge_symmetry,
        labels=['Vanilla', 'Symmetry Bias'],
        title='Convergence Time',
        xlabel='Model',
        ylabel='# of Steps to Converge',
        save_path=save_path
    )

    save_path = save_dir + 'stat_test_result.txt'
    run_statistical_tests(
        save_path=save_path,
        result_1=steps_to_converge_vanilla,
        result_2=steps_to_converge_symmetry,
        is_pair=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=100)
    parser.add_argument(
        '--save_dir', type=str,
        default='/Users/shiro/research/projects/symmetry-bias/results/'
                        )
    args = parser.parse_args()
    main(args)
