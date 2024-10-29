import json
import numpy as np
import csv
import argparse
from copy import deepcopy
import os
from pathlib import Path
from cryptorandom.cryptorandom import SHA256, int_from_hash

from shangrla.core.Audit import Audit, Assertion, Contest, CVR
from shangrla.core.NonnegMean import NonnegMean

margins = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]
erates = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]


def main(error_rate=0.03, margin=0.03, size=50_000):
    # Create CVR data file
    n_mismatch = int(size * error_rate)
    cvr_input = []
    for i in range(size - n_mismatch):
        cvr_input.append({"id": "match"+str(i+1), "votes": {"1": {"match": 1, "mismatch": 0}}})
    for i in range(n_mismatch):
        cvr_input.append({"id": "mismatch"+str(i+1), "votes": {"1": {"match": 0, "mismatch": 1}}})

    # print(f'Read {len(cvr_input)} rows')
    # cvr_input = np.array(cvr_input)

    cvr_list = CVR.from_dict(cvr_input)
    max_cards = len(cvr_list)
    # print(f'After merging, there are CVRs for {max_cards} cards')

    # Set specifications for the audit.
    audit = Audit.from_dict({
        'strata' : {'stratum1' : {'use_style'   : True,
                                  'replacement' : False}
            }
        })
    contest_dict = {'1':{'name'             : '1',
                         'risk_limit'       : 0.000001,
                         'cards'            : max_cards,
                         'choice_function'  : Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                         'n_winners'        : 1,
                         'share_to_win'     : 1-margin,
                         'candidates'       : ['match', 'mismatch'],
                         'winner'           : ['match'],
                         'audit_type'       : Audit.AUDIT_TYPE.POLLING,
                         'test'             : NonnegMean.alpha_mart,
                         'estim'            : NonnegMean.shrink_trunc
                         }
                    }
    contests = Contest.from_dict_of_dicts(contest_dict)

    # Construct the dict of dicts of assertions for each contest.
    Assertion.make_all_assertions(contests)
    audit.check_audit_parameters(contests)

    # Calculate margins for each assertion.
    min_margin = Assertion.set_all_margins_from_cvrs(audit, contests, cvr_list)
    #
    # # Some output.
    # print(f"Smallest margin in contest: {min_margin}")
    # print("Assorters and margins:")
    Contest.print_margins(contests)
    # print(audit.find_sample_size(contests, cvrs=cvr_list))
    #prng = SHA256(1)
    #CVR.assign_sample_nums(cvr_list, prng)

    # Calculate all of the p-values.
    pvalue_histories_array = calc_pvalues_all_orderings(contests, cvr_input)
    for pvalue_history in pvalue_histories_array:
        below_10pct = pvalue_history <= 0.10
        below_5pct = pvalue_history <= 0.05
        below_1pct = pvalue_history <= 0.01
        certified_10pct = True
        certified_5pct = True
        certified_1pct = True
        where_10pct = np.argmax(below_10pct) + 1
        where_5pct = np.argmax(below_5pct) + 1
        where_1pct = np.argmax(below_1pct) + 1
        if sum(below_10pct) == 0 or pvalue_history[where_10pct-1] == 0.0:
            certified_10pct = False
            where_10pct = len(pvalue_history)
        if sum(below_5pct) == 0 or pvalue_history[where_5pct-1] == 0.0:
            certified_5pct = False
            where_5pct = len(pvalue_history)
        if sum(below_1pct) == 0 or pvalue_history[where_1pct-1] == 0.0:
            certified_1pct = False
            where_1pct = len(pvalue_history)
        print(f"{size}, {error_rate}, {margin}, {min_margin}, {where_10pct}, {where_5pct}, {where_1pct}, "
              f"{certified_10pct}, {certified_5pct}, {certified_1pct}")


# =============================================================================
# Define functions.

# Shuffle raw RAIRE-formatted CVRs according to a given ordering.
#
# Need to concatenate [0, 1] at the start, to avoid shuffling the file headers.
# The other indices need to be translated accordingly (by +1).
def shuffle(cvrs, ordering):
    neworder = np.concatenate(([0, 1], ordering + 1))
    cvrs_shuffled = [cvrs[i] for i in neworder]
    return cvrs_shuffled


# Extract a 'merged' p-value history, combined across all assertions.
#
# This returns a single list of p-values, which is the p-value at each stage of
# sampling.  That is, the smallest value of `alpha` for which the audit may
# terminate (and certify) at that stage.
def merge_pvalues(assertions_dict):
    pvalue_histories = []
    for asrtn in assertions_dict:
        a = assertions_dict[asrtn]
        phist = a.p_history
        phist_running_min = np.minimum.accumulate(phist)
        pvalue_histories.append(phist_running_min)
    pvalue_histories_stacked = np.stack(pvalue_histories)
    pvalue_histories_merged  = np.amax(pvalue_histories_stacked, axis=0)
    return pvalue_histories_merged


# Calculate p-values for a given ordering.
def calc_pvalues_single_ordering(contests, cvr_input, rng, ordering_index):
    #print("Working on ordering {}".format(ordering_index))

    # Shuffle ballots according to the given ordering.
    cvr_input_shuffled = deepcopy(cvr_input)
    rng.shuffle(cvr_input_shuffled)

    # Import shuffled CVRs.
    shuffled_ballots = CVR.from_dict(cvr_input_shuffled)

    # Find measured risks for all assertions.
    Assertion.set_p_values(contests, shuffled_ballots, None)

    # Extract all of the p-value histories and combine them.
    pvalues = merge_pvalues(contests['1'].assertions)

    return pvalues


# Calculate p-values for a set of orderings.
def calc_pvalues_all_orderings(contests, cvr_input, n_orderings=1000):
    pvalue_list = []
    rng = np.random.default_rng(seed=2024_10_29)
    for o in range(n_orderings):
        pvalue_list.append(calc_pvalues_single_ordering(contests, cvr_input, rng, o))
    # pvalue_array = np.stack(pvalue_list, axis=-1)
    return pvalue_list


if __name__ == "__main__":
    counter = 0  # 1-13
    for erate in erates:
        counter += 1
        if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue
        print("pop_size, error_rate, margin, assertion_margin, sample_size_10pct, sample_size_5pct, sample_size_1pct, "
          "certified_10pct, certified_5pct, certified_1pct")
        for margin in margins:
            main(error_rate=erate, margin=margin)
