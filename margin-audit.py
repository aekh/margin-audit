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

margins = [0.0001, 0.0005, 0.0010, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500,
           0.0600, 0.0700, 0.0800, 0.0900, 0.1000, 0.1250, 0.1500]
erates = [0.0000, 0.0001, 0.0005, 0.0010, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500]


def main(error_rate=0.03, margin=0.03, size=50_000):
    # Create CVR data file
    n_mismatch = int(size * error_rate)
    assert n_mismatch / size == error_rate, "Given error pct not attainable"  # Ensure there is no rounding error
    ballot_margin = int(size * margin)
    assert ballot_margin / size == margin, "Given margin pct not attainable"  # Ensure there is no rounding error

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
    counter = 0  # array size on SLURM cluster == 1-13
    for margin in margins:
        counter += 1
        if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue  # parallelise on SLURM cluster
        print("pop_size, error_rate, margin, assertion_margin, sample_size_10pct, sample_size_5pct, sample_size_1pct, "
          "certified_10pct, certified_5pct, certified_1pct")
        for erate in erates:
            main(error_rate=erate, margin=margin)
