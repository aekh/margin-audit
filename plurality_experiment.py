import itertools
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


def main(pop_size, margin, erate, dist, pct_blank, n_runs):
    audit = Audit.from_dict({
        'strata': {'stratum1': {'use_style': True,
                                'replacement': False}}
    })

    ballot_margin = int(round(pop_size * margin))
    assert ballot_margin % 2 == 0, "UNSUPPORTED: margin is odd number of ballots!"

    n_errors = int(round(pop_size * erate))
    n_invalid = int(round(pop_size * pct_blank))

    mb_contests, mb_cvr_input = create_marginbased_audit(pop_size, margin)
    cc_contests, cc_cvr_input = create_card_comp_audit(ballot_margin, n_invalid, pop_size)

    # Construct the dict of dicts of assertions for each contest.
    Assertion.make_all_assertions(mb_contests)
    Assertion.make_all_assertions(cc_contests)
    audit.check_audit_parameters(mb_contests)
    audit.check_audit_parameters(cc_contests)

    # Calculate margins for each assertion.
    mb_cvr_list = CVR.from_dict(mb_cvr_input)
    cc_cvr_list = CVR.from_dict(cc_cvr_input)
    mb_min_margin = Assertion.set_all_margins_from_cvrs(audit, mb_contests, mb_cvr_list)
    cc_min_margin = Assertion.set_all_margins_from_cvrs(audit, cc_contests, cc_cvr_list)

    # print(f"  {mb_min_margin=}")

    # Hack, set attributes for tests and estims
    for contest in mb_contests.values():
        for assertion in contest.assertions.values():
            assertion.test.error_rate_2 = 1e-5
            assertion.test.eta = assertion.test.u * (1 - 0.001)
            # assertion.test.d = d_value
    for contest in cc_contests.values():
        for assertion in contest.assertions.values():
            assertion.test.error_rate_2 = 1e-5

    # Calculate all of the p-values.
    if erate == 0.0:
        n_runs = 1
    for i in range(n_runs):
        rng = np.random.default_rng(seed=2025_01_08+i)
        mb_pvalues_100, mb_pvalues_inf, mb_overstatements, cc_pvalues, cc_overstatements = calc_pvalues_single_ordering(
            mb_contests, mb_cvr_input, cc_contests, cc_cvr_input, pop_size, rng, n_errors, dist)
        mb_samplesize_5pct_100, mb_certified_5pct_100, mb_counts_5pct_100 = get_samplesize_and_overstatements(
            mb_overstatements, mb_pvalues_100, 0.05)
        mb_samplesize_1pct_100, mb_certified_1pct_100, mb_counts_1pct_100 = get_samplesize_and_overstatements(
            mb_overstatements, mb_pvalues_100, 0.01)
        mb_samplesize_5pct_inf, mb_certified_5pct_inf, mb_counts_5pct_inf = get_samplesize_and_overstatements(
            mb_overstatements, mb_pvalues_inf, 0.05)
        mb_samplesize_1pct_inf, mb_certified_1pct_inf, mb_counts_1pct_inf = get_samplesize_and_overstatements(
            mb_overstatements, mb_pvalues_inf, 0.01)
        cc_samplesize_5pct, cc_certified_5pct, cc_counts_5pc = get_samplesize_and_overstatements(
            cc_overstatements, cc_pvalues, 0.05)
        cc_samplesize_1pct, cc_certified_1pct, cc_counts_1pc = get_samplesize_and_overstatements(
            cc_overstatements, cc_pvalues, 0.01)

        print(f"ballotmargin, {pop_size}, {margin}, {pct_blank}, {dist}, {erate}, {mb_min_margin}, 100, 0.05, "
              f"{mb_samplesize_5pct_100}, {mb_certified_5pct_100}, "
              f"{mb_counts_5pct_100[-2]}, {mb_counts_5pct_100[-1]}, {mb_counts_5pct_100[1]}, {mb_counts_5pct_100[2]}")
        print(f"ballotmargin, {pop_size}, {margin}, {pct_blank}, {dist}, {erate}, {mb_min_margin}, 100, 0.01, "
              f"{mb_samplesize_1pct_100}, {mb_certified_1pct_100}, "
              f"{mb_counts_1pct_100[-2]}, {mb_counts_1pct_100[-1]}, {mb_counts_1pct_100[1]}, {mb_counts_1pct_100[2]}")
        print(f"ballotmargin, {pop_size}, {margin}, {pct_blank}, {dist}, {erate}, {mb_min_margin}, 10xPop, 0.05, "
              f"{mb_samplesize_5pct_inf}, {mb_certified_5pct_inf}, "
              f"{mb_counts_5pct_inf[-2]}, {mb_counts_5pct_inf[-1]}, {mb_counts_5pct_inf[1]}, {mb_counts_5pct_inf[2]}")
        print(f"ballotmargin, {pop_size}, {margin}, {pct_blank}, {dist}, {erate}, {mb_min_margin}, 10xPop, 0.01, "
              f"{mb_samplesize_1pct_inf}, {mb_certified_1pct_inf}, "
              f"{mb_counts_1pct_inf[-2]}, {mb_counts_1pct_inf[-1]}, {mb_counts_1pct_inf[1]}, {mb_counts_1pct_inf[2]}")
        print(f"cardcomparison, {pop_size}, {margin}, {pct_blank}, {dist}, {erate}, {cc_min_margin}, NA, 0.05, "
              f"{cc_samplesize_5pct}, {cc_certified_5pct}, "
              f"{cc_counts_5pc[-2]}, {cc_counts_5pc[-1]}, {cc_counts_5pc[1]}, {cc_counts_5pc[2]}")
        print(f"cardcomparison, {pop_size}, {margin}, {pct_blank}, {dist}, {erate}, {cc_min_margin}, NA, 0.01, "
              f"{cc_samplesize_1pct}, {cc_certified_1pct}, "
              f"{cc_counts_1pc[-2]}, {cc_counts_1pc[-1]}, {cc_counts_1pc[1]}, {cc_counts_1pc[2]}")


def get_samplesize_and_overstatements(mb_overstatements, mb_pvalues, pvalue_threshold):
    mb_below_thres = mb_pvalues <= pvalue_threshold
    mb_certified_thres = True
    mb_where_thres = np.argmax(mb_below_thres) + 1
    if sum(mb_below_thres) == 0 or mb_pvalues[mb_where_thres - 1] == 0.0:
        mb_certified_thres = False
        mb_where_thres = len(mb_pvalues)
    mb_type_thres, mb_counts_thres = np.unique(mb_overstatements[:mb_where_thres], return_counts=True)
    mb_counts_thres = dict(zip([int(x) for x in mb_type_thres], [int(x) for x in mb_counts_thres]))
    mb_counts_thres.setdefault(-2, 0)
    mb_counts_thres.setdefault(-1, 0)
    mb_counts_thres.setdefault(0, 0)
    mb_counts_thres.setdefault(1, 0)
    mb_counts_thres.setdefault(2, 0)
    sample_size = mb_where_thres
    is_certified = mb_certified_thres
    overstatement_counts = mb_counts_thres
    return sample_size, is_certified, overstatement_counts


def create_card_comp_audit(ballot_margin, n_invalid, pop_size):
    n_valid = pop_size - n_invalid
    assert n_valid % 2 == 0, "UNSUPPORTED: number of valid ballots is odd!"
    n_alice = (n_valid // 2) + ballot_margin
    n_bob = (n_valid // 2) - ballot_margin
    assert n_alice > n_bob and n_alice + n_bob == n_valid, f"ERROR: invalid number of ballots! {n_alice=}, {n_bob=}"
    assert n_alice > 0 and n_bob >= 0, f"ERROR: ballots are negative! {n_alice=}, {n_bob=}"
    cvr_input = []
    for i in range(n_alice):
        cvr_input.append({"id": "alice" + str(i + 1), "votes": {"1": {"alice": 1, "bob": 0}}})
    for i in range(n_bob):
        cvr_input.append({"id": "bob" + str(i + 1), "votes": {"1": {"alice": 0, "bob": 1}}})
    for i in range(n_invalid):
        cvr_input.append({"id": "blank" + str(i + 1), "votes": {"1": {"alice": 0, "bob": 0}}})
    contest_dict = {'1': {'name': '1',
                          'risk_limit': 0.05,
                          'cards': pop_size,
                          'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                          'n_winners': 1,
                          'share_to_win': 1 / 2,
                          'candidates': ['alice', 'bob'],
                          'winner': ['alice'],
                          'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                          'test': NonnegMean.alpha_mart,
                          'estim': NonnegMean.optimal_comparison,
                          }
                    }
    contests = Contest.from_dict_of_dicts(contest_dict)
    return contests, cvr_input


def create_marginbased_audit(pop_size, margin):
    cvr_input = []
    for i in range(pop_size):
        cvr_input.append({"id": "match" + str(i + 1), "votes": {"1": {"match": 1, "mismatch": 0}}})
    contest_dict = {'1': {'name': '1',
                          'risk_limit': 0.05,
                          'cards': pop_size,
                          'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                          'n_winners': 1,
                          'share_to_win': 1 - margin,
                          'candidates': ['match', 'mismatch'],
                          'winner': ['match'],
                          'audit_type': Audit.AUDIT_TYPE.POLLING,
                          'test': NonnegMean.alpha_mart,
                          'estim': NonnegMean.shrink_trunc,
                          }
                    }
    contests = Contest.from_dict_of_dicts(contest_dict)
    return contests, cvr_input


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


def shuffle(cvrs, ordering):
    cvrs_shuffled = [cvrs[i-1] for i in ordering]
    return cvrs_shuffled


# Calculate p-values for a given ordering.
def calc_pvalues_single_ordering(mb_contests, mb_cvr_input, cc_contests, cc_cvr_input, pop_size, rng, n_errors, dist):
    mb_mvr_input = deepcopy(mb_cvr_input)
    cc_mvr_input = deepcopy(cc_cvr_input)
    if dist == "all_2_under":
        vote_ids = [i for i, ballot in enumerate(cc_cvr_input) if ballot['votes']['1']['bob'] == 1]
        error_ids = rng.choice(vote_ids, n_errors, replace=False, shuffle=False)  # sample votes for bob
        for idx in error_ids:
            id = cc_mvr_input[idx]['id']
            cc_mvr_input[idx] = {"id": id, "votes": {"1": {"alice": 1, "bob": 0}}}
    elif dist == "all_2_over":
        vote_ids = [i for i, ballot in enumerate(cc_cvr_input) if ballot['votes']['1']['alice'] == 1]
        error_ids = rng.choice(vote_ids, n_errors, replace=False, shuffle=False)  # sample votes for alice
        for idx in error_ids:
            id = cc_mvr_input[idx]['id']
            cc_mvr_input[idx] = {"id": id, "votes": {"1": {"alice": 0, "bob": 1}}}
    else:  # dist == "random"
        error_ids = rng.choice(range(len(cc_cvr_input)), n_errors, replace=False, shuffle=False)  # sample random votes
        for idx in error_ids:
            id = cc_mvr_input[idx]['id']
            if cc_cvr_input[idx]['votes']['1']['alice'] == 1:
                change_to = rng.choice([{"alice": 0, "bob": 1}, {"alice": 0, "bob": 0}])
                cc_mvr_input[idx] = {"id": id, "votes": {"1": change_to}}
            elif cc_cvr_input[idx]['votes']['1']['bob'] == 1:
                change_to = rng.choice([{"alice": 1, "bob": 0}, {"alice": 0, "bob": 0}])
                cc_mvr_input[idx] = {"id": id, "votes": {"1": change_to}}
            else:  # blank
                change_to = rng.choice([{"alice": 1, "bob": 0}, {"alice": 0, "bob": 1}, {"alice": 0, "bob": 0}])
                cc_mvr_input[idx] = {"id": id, "votes": {"1": change_to}}

    # introduce errors in margin-based audit
    for idx in error_ids:
        mb_mvr_input[idx] = {"id": "mismatch" + str(idx + 1), "votes": {"1": {"match": 0, "mismatch": 1}}}

    # create shuffle ordering
    ordering = rng.permutation(len(mb_mvr_input))
    shuffled_mb_mvr_input = shuffle(mb_mvr_input, ordering)
    shuffled_cc_cvr_input = shuffle(cc_cvr_input, ordering)
    shuffled_cc_mvr_input = shuffle(cc_mvr_input, ordering)

    # Import shuffled CVRs.
    shuffled_mb_mvrs = CVR.from_dict(shuffled_mb_mvr_input)
    shuffled_cc_cvrs = CVR.from_dict(shuffled_cc_cvr_input)
    shuffled_cc_mvrs = CVR.from_dict(shuffled_cc_mvr_input)

    Assertion.set_p_values(mb_contests, shuffled_mb_mvrs, None)
    mb_pvalues_100 = merge_pvalues(mb_contests['1'].assertions)
    mb_overstatements = np.array([0 if shuffled_mb_mvr_input[i]['votes']['1']['match'] == 1 else -2
                                  for i in range(len(shuffled_mb_mvr_input))])
    # print(f"mb_pvalues_100: {",".join(str(i) for i in mb_pvalues_100[:25])}")
    # print(f"mb_overstatements: {",".join(str(i) for i in mb_overstatements[:25])}")

    for contest in mb_contests.values():
        for assertion in contest.assertions.values():
            assertion.test.d = pop_size*10
    Assertion.set_p_values(mb_contests, shuffled_mb_mvrs, None)
    mb_pvalues_inf = merge_pvalues(mb_contests['1'].assertions)

    # print(f"mb_pvalues_inf: {",".join(str(i) for i in mb_pvalues_inf[:25])}")

    Assertion.set_p_values(cc_contests, shuffled_cc_mvrs, shuffled_cc_cvrs, use_all=True)
    cc_pvalues = merge_pvalues(cc_contests['1'].assertions)
    assertion = next(iter(cc_contests['1'].assertions.values()))
    cc_overstatements = np.array([assertion.assorter.overstatement(shuffled_cc_mvrs[i], shuffled_cc_cvrs[i])*2
                                  for i in range(len(shuffled_cc_cvrs))])

    # print(f"cc_pvalues: {",".join(str(i) for i in cc_pvalues[:25])}")
    # print(f"cc_overstatements: {",".join(str(i) for i in cc_overstatements[:25])}")

    return mb_pvalues_100, mb_pvalues_inf, mb_overstatements, cc_pvalues, cc_overstatements


if __name__ == "__main__":
    n_runs = 1_000
    pop_sizes = [10_000, 50_000, 100_000]
    margins = [1e-3, 2e-3, 3e-3, 6e-3,
               1e-2, 2e-2, 3e-2, 6e-2,
               1e-1, 2e-1, 3e-1]
    erates = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    dists = ["all_2_over", "all_2_under", "random"]
    pct_blanks = [0.0, 0.80]
    counter = 0  # 1-960
    for _pop_size, _margin, _erate, _dist, _pct_blank in itertools.product(
            pop_sizes, margins, erates, dists, pct_blanks):
        n_valid = int(round((1 - _pct_blank) * _pop_size))
        ballot_margin = int(round(_pop_size * _margin))
        if (n_valid // 2) - ballot_margin < 0:
            # print(f"Skipping {version}, {pop_size}, {margin}, {erate}, {d_value}, {dist}, {pct_blank}, "
            #       f"because {(n_valid // 2)} < {ballot_margin}")
            continue  # skip unfeasible margins
        if _erate == 0.0 and _dist != "random":
            continue  # skip overstatement distributions for zero error rates
        counter += 1
        # print(counter); continue
        if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue
        print("method, pop_size, margin, pct_blank, dist, error_rate, assorter_margin, d_value, alpha, "
              "sample_size, certified, n_two_over, n_one_over, n_one_under, n_two_under")
        # print("method, pop_size, margin, error_rate, d_value, dist, pct_blank, sample_size_5pct, sample_size_1pct, "
        #       "certified_5pct, certified_1pct, n_two_over, n_one_over, n_one_under, n_two_under")
        main(_pop_size, _margin, _erate, _dist, _pct_blank, n_runs)
