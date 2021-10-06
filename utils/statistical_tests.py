"""
Function to run statistical test
"""
from scipy import stats
from typing import List


def run_statistical_tests(save_path: str, result_1: List, result_2: List,
                          p_thr=0.05, is_pair=False) -> None:
    """Run following statistical hypothesis tests and save the result as text file:
       1. Shapiro-Wilk Test: Gaussian or not
       2. F Test: equal variance or not
       3. Paired student's t-test: equal mean or not
       4. Student's t-test: equal mean or not
       5. Welch's t-test: equal mean or not
       6. Wilcoxon Signed-Rank Test:equal median or not
       7. Mann-Whitney U Test: equal median or not

    Args:
        save_path (str): Path to save the result text file.
        result_1 (List): Result to be used for test.
        result_2 (List): Result to be used for test.
        p_thr (float, optional): Thresohold of p value. Defaults to 0.05.
        is_pair (bool, optional): If a set of two results is a pair or not.
                                  Defaults to False.
    """
    with open(save_path, 'w') as f:
        is_gaussian = False
        # Shapiro-Wilk Test: Gaussian or not
        stat_1, p_val_1 = stats.shapiro(result_1)
        stat_2, p_val_2 = stats.shapiro(result_2)
        outcome = \
            f'Shapiro-Wilk Test 1: stat={stat_1:.3f}, p={p_val_1:.3f}'
        print(outcome, file=f)
        outcome = \
            f'Shapiro-Wilk Test 2: stat={stat_2:.3f}, p={p_val_2:.3f}'
        print(outcome, file=f)
        if p_val_1 > p_thr and p_val_2 > p_thr:
            is_gaussian = True
            print('Failed to reject hypothesis of Gaussianity\n', file=f)
        else:
            print('Reject hypothesis of Gaussianity\n', file=f)

        if is_gaussian:
            # F Test: equal variance or not
            n_1 = len(result_1)
            n_2 = len(result_2)
            var_1 = stats.tvar(result_1)
            var_2 = stats.tvar(result_2)
            dfn = n_2 - 1
            dfd = n_1 - 1
            f_val = var_2 / var_1
            p_val = 1 - stats.f.cdf(f_val, dfn=dfn, dfd=dfd)
            outcome = f'F Test: f_val={f_val:.3f}, p={p_val:.3f}'
            print(outcome, file=f)
            if p_val > p_thr:
                is_equal_variance = True
                print('Failed to reject hypothesis of equal variance\n', file=f)
            else:
                is_equal_variance = False
                print('Reject hypothesis of equal variance\n', file=f)

            if is_pair:
                # Paired student's t-test: equal mean or not
                if is_equal_variance:
                    stat, p_val = stats.ttest_rel(
                        result_1, result_2)
                    outcome = \
                        f'Paired Student’s t-test: stat={stat:.3f}, p={p_val:.3f}'
                    print(outcome, file=f)
                    if p_val > p_thr:
                        print('Failed to reject hypothesis of equal mean\n',
                              file=f)
                        print('Difference is not significant', file=f)
                    else:
                        print('Reject hypothesis of equal mean\n', file=f)
                        print('Difference is significant', file=f)
                else:
                    # Wilcoxon Signed-Rank Test:equal median or not
                    stat, p_val = stats.wilcoxon(result_1, result_2)
                    outcome = \
                        f'Wilcoxon Signed-Rank Test: stat={stat:.3f}, p={p_val:.3f}'
                    print(outcome, file=f)
                    if p_val > p_thr:
                        print('Failed to reject hypothesis of equal median\n',
                              file=f)
                        print('Difference is not significant', file=f)
                    else:
                        print('Reject hypothesis of equal median\n', file=f)
                        print('Difference is significant', file=f)

            else:
                # Student's t-test / Welch's t-test: equal mean or not
                stat, p_val = stats.ttest_ind(
                    result_1, result_2, equal_var=is_equal_variance)
                outcome = f'Student’s t-test: stat={stat:.3f}, p={p_val:.3f}'
                print(outcome)
                if p_val > p_thr:
                    print('Failed to reject hypothesis of equal mean\n',
                          file=f)
                    print('Difference is not significant', file=f)
                else:
                    print('Reject hypothesis of equal mean\n', file=f)
                    print('Difference is significant', file=f)
        else:
            if is_pair:
                # Wilcoxon Signed-Rank Test: equal median or not
                stat, p_val = stats.wilcoxon(result_1, result_2)
                outcome = \
                    f'Wilcoxon Signed-Rank Test: stat={stat:.3f}, p={p_val:.3f}'
                print(outcome, file=f)
                if p_val > p_thr:
                    print('Failed to reject hypothesis of equal median\n',
                          file=f)
                    print('Difference is not significant', file=f)
                else:
                    print('Reject hypothesis of equal median\n', file=f)
                    print('Difference is significant', file=f)
            else:
                # Mann-Whitney U Test: equal median or not
                stat, p_val = stats.mannwhitneyu(result_1, result_2)
                outcome = \
                    f'Mann-Whitney U Test: stat={stat:.3f}, p={p_val:.3f}'
                print(outcome, file=f)
                if p_val > p_thr:
                    print('Failed to reject hypothesis of equal median\n',
                          file=f)
                    print('Difference is not significant', file=f)
                else:
                    print('Reject hypothesis of equal median\n', file=f)
                    print('Difference is significant', file=f)
