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

from raire_utils import load_contests_from_raire_raw
from raire import compute_raire_assertions
from sample_estimator import cp_estimate

import itertools


def read_election_files(datafile, marginfile, orderfile):
    rairedata_pre = "1\nContest,"
    rairedata = ""

    ballotnmbr = 0
    with open(datafile, "r") as file:
        line = file.readline()
        candmap = dict()
        candlist = [i.strip() for i in line.split(",")]
        for i, cand in enumerate(candlist):
            candmap[cand] = i
        ncand = len(candlist)

        # remove headers
        while True:
            if "-" in file.readline():
                break

        for line in file:
            f = line.split(" : ")
            strballot = f[0].split("(")[1].split(")")[0].split(",")
            if len(strballot) == 2 and strballot[1] == '':
                strballot = [strballot[0]]  # compatibility issue fix with trailing comma
            if len(strballot) == 1 and strballot[0] == '':
                strballot = []
            ballot = [candmap[i.strip()] for i in strballot]
            nvotes = int(f[1])
            for _ in range(nvotes):
                ballotnmbr += 1
                seq = ",".join([str(i) for i in ballot])
                rairedata += f"\n1,{ballotnmbr},{seq}"
    nballots = ballotnmbr

    margindata = [None] * ncand
    try:
        with open(marginfile, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if csv_reader.line_num == 1: continue
                margindata[candmap[row[1].strip()]] = int(row[2]) / nballots
    except FileNotFoundError:
        pass
    margin = max(margindata, default=None)
    winner = int(np.argmax(np.array(margindata)))

    orderdata = []
    with open(orderfile, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if csv_reader.line_num == 1: continue
            orderdata.append([int(i) for i in row])

    rairedata_pre += f"1,{ncand}," + ",".join([str(i) for i in range(ncand)]) + f",winner,{winner}"
    rairedata = rairedata_pre + rairedata

    return ncand, winner, nballots, margin, orderdata, rairedata


def main(name, data, pop_size, ncand, winner, margin, orderdata, erate, dist, n_runs, runset=None):
    audit = Audit.from_dict({
        'strata': {'stratum1': {'use_style': True,
                                'replacement': False}}
    })

    ballot_margin = int(round(pop_size * margin))
    # assert ballot_margin % 2 == 0, "UNSUPPORTED: margin is odd number of ballots!"

    n_errors = int(round(pop_size * erate))

    mb_contests, mb_cvr_input = create_marginbased_audit(pop_size, margin)
    cc_contests, cc_cvr_input = create_card_comp_audit(data, ncand, winner, pop_size)

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

    hardest_assertion_name, hardest_assertion = min(cc_contests['1'].assertions.items(), key=lambda x: x[1].margin)

    # Hack, set attributes for tests and estims
    for contest in mb_contests.values():
        for assertion in contest.assertions.values():
            assertion.test.error_rate_2 = 1e-5
            assertion.test.eta = assertion.test.u * (1 - 0.001)
            # assertion.test.d = d_value
    for contest in cc_contests.values():
        for assertion in contest.assertions.values():
            assertion.test.error_rate_2 = 1e-5

    # Calculate all the p-values.
    if erate == 0.0:
        n_runs = 1
    if runset is None:
        runset = range(n_runs)
    for i in runset:
        rng = np.random.default_rng(seed=2025_01_16+i)
        mb_pvalues_100, mb_pvalues_inf, mb_overstatements, cc_pvalues, cc_overstatements = calc_pvalues_single_ordering(
            mb_contests, mb_cvr_input, cc_contests, cc_cvr_input, pop_size, rng, n_errors, dist,
            hardest_assertion, orderdata, i)

        if dist == "random":
            mb_samplesize_5pct_100, mb_certified_5pct_100, mb_counts_5pct_100 = get_samplesize_and_overstatements(
                mb_overstatements, mb_pvalues_100, 0.05)
            mb_samplesize_5pct_inf, mb_certified_5pct_inf, mb_counts_5pct_inf = get_samplesize_and_overstatements(
                mb_overstatements, mb_pvalues_inf, 0.05)
            print(f"{name}, ballotmargin, {pop_size}, {margin}, NA, {erate}, {mb_min_margin}, 100, 0.05, "
                  f"{i}, {mb_samplesize_5pct_100}, {mb_certified_5pct_100}, "
                  f"{mb_counts_5pct_100[2]}, {mb_counts_5pct_100[1]}, {mb_counts_5pct_100[-1]}, {mb_counts_5pct_100[-2]}")
            print(f"{name}, ballotmargin, {pop_size}, {margin}, NA, {erate}, {mb_min_margin}, 10xPop, 0.05, "
                  f"{i}, {mb_samplesize_5pct_inf}, {mb_certified_5pct_inf}, "
                  f"{mb_counts_5pct_inf[2]}, {mb_counts_5pct_inf[1]}, {mb_counts_5pct_inf[-1]}, {mb_counts_5pct_inf[-2]}")

        cc_samplesize_5pct, cc_certified_5pct, cc_counts_5pc = get_samplesize_and_overstatements(
            cc_overstatements, cc_pvalues, 0.05)

        print(f"{name}, cardcomparison, {pop_size}, {margin}, {dist}, {erate}, {cc_min_margin}, NA, 0.05, "
              f"{i}, {cc_samplesize_5pct}, {cc_certified_5pct}, "
              f"{cc_counts_5pc[2]}, {cc_counts_5pc[1]}, {cc_counts_5pc[-1]}, {cc_counts_5pc[-2]}")


def get_samplesize_and_overstatements(overstatements, pvalues, pvalue_threshold):
    below_thres = pvalues <= pvalue_threshold
    certified_thres = True
    where_thres = np.argmax(below_thres) + 1
    if sum(below_thres) == 0 or pvalues[where_thres - 1] == 0.0:
        certified_thres = False
        where_thres = len(pvalues)
    type_thres, counts_thres = np.unique(overstatements[:where_thres], return_counts=True)
    counts_thres = dict(zip([int(x) for x in type_thres], [int(x) for x in counts_thres]))
    counts_thres.setdefault(-2, 0)
    counts_thres.setdefault(-1, 0)
    counts_thres.setdefault(0, 0)
    counts_thres.setdefault(1, 0)
    counts_thres.setdefault(2, 0)
    sample_size = where_thres
    is_certified = certified_thres
    overstatement_counts = counts_thres
    return sample_size, is_certified, overstatement_counts


def create_card_comp_audit(data, ncand, winner, pop_size):
    raire_contests, cvrs = load_contests_from_raire_raw(data)
    raire_contest = raire_contests[0]
    cvr_input = [{"id": c, "votes": {'1': {i: cvrs[c]['1'][i]+1 for i in cvrs[c]['1'].keys()}}} for c in cvrs.keys()]
    assertions = compute_raire_assertions(raire_contest, cvrs, str(winner), cp_estimate, log=False)
    assertions = [a.to_json() for a in assertions]

    contest_dict = {'1': {'name': '1',
                          'risk_limit': 0.05,
                          'cards': pop_size,
                          'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.IRV,
                          'n_winners': 1,
                          'candidates': [str(i) for i in range(ncand)],
                          'winner': [str(winner)],
                          'assertion_file': "./assertions_temp.json",
                          'assertion_json': assertions,
                          'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                          'test': NonnegMean.alpha_mart,
                          'estim': NonnegMean.optimal_comparison
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
def calc_pvalues_single_ordering(mb_contests, mb_cvr_input, cc_contests, cc_cvr_input, pop_size, rng, n_errors, dist,
                                 hardest_assertion, orderdata, ordering_index):
    cc_mvr_input = deepcopy(cc_cvr_input)
    asrt_loser = hardest_assertion.loser
    asrt_winner = hardest_assertion.winner
    ncand = len(cc_contests['1'].candidates)
    error_ids = rng.choice(range(len(cc_cvr_input)), n_errors, replace=False, shuffle=False)
    if dist == "all_under":  # all understatements
        for idx in error_ids:
            if cc_cvr_input[idx]['votes']['1'] == {asrt_winner: 1}:
                cc_mvr_input[idx]['votes']['1'] = {asrt_winner: 1, asrt_loser: 2}
            else:
                cc_mvr_input[idx]['votes']['1'] = {asrt_winner: 1}
    elif dist == "all_over":  # all overstatements
        for idx in error_ids:
            if cc_cvr_input[idx]['votes']['1'] == {asrt_loser: 1}:
                cc_mvr_input[idx]['votes']['1'] = {asrt_loser: 1, asrt_winner: 2}
            else:
                cc_mvr_input[idx]['votes']['1'] = {asrt_loser: 1}
    elif dist == "truncate":  # truncate ballots
        for idx in error_ids:
            votes = cc_cvr_input[idx]['votes']['1']
            nprefs = len(votes)
            if nprefs == 0:  # add a random candidate when ballot is empty
                add = rng.integers(0, ncand)
                cc_mvr_input[idx]['votes']['1'] = {str(add): 1}
            elif nprefs == 1:  # remove all votes when ballot has only one preference
                cc_mvr_input[idx]['votes']['1'] = {}
            else:  # cut ballot at random position
                cut_at = rng.integers(0, nprefs-1)
                cc_mvr_input[idx]['votes']['1'] = {cand: votes[cand] for cand in votes.keys() if votes[cand] <= cut_at}
    else:  # dist == "random"
        for idx in error_ids:
            votes = cc_cvr_input[idx]['votes']['1']
            while True:
                ballot_length = len(rng.choice(cc_cvr_input)['votes']['1'])  # randomly sample from CVR to get length
                newcands = rng.choice(range(ncand), ballot_length, replace=False, shuffle=False)
                newcands = rng.permutation(newcands)
                newvote = {str(cand): i+1 for i, cand in enumerate(newcands)}
                if newvote != votes:
                    break  # make sure new vote is different from original vote
            cc_mvr_input[idx]['votes']['1'] = newvote

    # introduce errors in margin-based audit
    if dist == "random":
        mb_mvr_input = deepcopy(mb_cvr_input)
        for idx in error_ids:
            mb_mvr_input[idx] = {"id": "mismatch" + str(idx + 1), "votes": {"1": {"match": 0, "mismatch": 1}}}

    # create shuffle ordering
    ordering = orderdata[ordering_index]
    if dist == "random":
        shuffled_mb_mvr_input = shuffle(mb_mvr_input, ordering)
    shuffled_cc_cvr_input = shuffle(cc_cvr_input, ordering)
    shuffled_cc_mvr_input = shuffle(cc_mvr_input, ordering)

    # Import shuffled CVRs.
    if dist == "random":
        shuffled_mb_mvrs = CVR.from_dict(shuffled_mb_mvr_input)
    shuffled_cc_cvrs = CVR.from_dict(shuffled_cc_cvr_input)
    shuffled_cc_mvrs = CVR.from_dict(shuffled_cc_mvr_input)

    if dist == "random":
        Assertion.set_p_values(mb_contests, shuffled_mb_mvrs, None)
        mb_pvalues_100 = merge_pvalues(mb_contests['1'].assertions)
        mb_overstatements = np.array([0 if shuffled_mb_mvr_input[i]['votes']['1']['match'] == 1 else 2
                                      for i in range(len(shuffled_mb_mvr_input))])

        for contest in mb_contests.values():
            for assertion in contest.assertions.values():
                assertion.test.d = pop_size*10
        Assertion.set_p_values(mb_contests, shuffled_mb_mvrs, None)
        mb_pvalues_inf = merge_pvalues(mb_contests['1'].assertions)
    else:
        mb_pvalues_100 = None
        mb_pvalues_inf = None
        mb_overstatements = None

    Assertion.set_p_values(cc_contests, shuffled_cc_mvrs, shuffled_cc_cvrs, use_all=True)
    cc_pvalues = merge_pvalues(cc_contests['1'].assertions)
    cc_overstatements = np.array([hardest_assertion.assorter.overstatement(shuffled_cc_mvrs[i], shuffled_cc_cvrs[i])*2
                                  for i in range(len(shuffled_cc_cvrs))])

    # print(f"cc_pvalues: {",".join(str(i) for i in cc_pvalues[:25])}")
    # print(f"cc_overstatements: {",".join(str(i) for i in cc_overstatements[:25])}")

    return mb_pvalues_100, mb_pvalues_inf, mb_overstatements, cc_pvalues, cc_overstatements


datafiles_nsw = np.array(
    ["Albury","Auburn","Ballina","Balmain","Bankstown","Barwon","Bathurst","Baulkham_Hills","Bega","Blacktown",
     "Blue_Mountains","Cabramatta","Camden","Campbelltown","Canterbury","Castle_Hill","Cessnock","Charlestown",
     "Clarence","Coffs_Harbour","Coogee","Cootamundra","Cronulla","Davidson","Drummoyne","Dubbo","East_Hills",
     "Epping","Fairfield","Gosford","Goulburn","Granville","Hawkesbury","Heathcote","Heffron","Holsworthy","Hornsby",
     "Keira","Kiama","Kogarah","Ku-ring-gai","Lake_Macquarie","Lakemba","Lane_Cove","Lismore","Liverpool",
     "Londonderry","Macquarie_Fields","Maitland","Manly","Maroubra","Miranda","Monaro","Mount_Druitt","Mulgoa",
     "Murray","Myall_Lakes","Newcastle","Newtown","Northern_Tablelands","North_Shore","Oatley","Orange","Oxley",
     "Parramatta","Penrith","Pittwater","Port_Macquarie","Port_Stephens","Prospect","Riverstone","Rockdale","Ryde",
     "Seven_Hills","Shellharbour","South_Coast","Strathfield","Summer_Hill","Swansea","Sydney","Tamworth","Terrigal",
     "The_Entrance","Tweed","Upper_Hunter","Vaucluse","Wagga_Wagga","Wakehurst","Wallsend","Willoughby",
     "Wollondilly","Wollongong","Wyong"])
datafiles_usirv = np.array(
    ["Aspen_2009_CityCouncil","Berkeley_2010_D1CityCouncil","Berkeley_2010_D7CityCouncil",
     "Oakland_2010_D4CityCouncil","Oakland_2010_Mayor","Pierce_2008_CountyAssessor","Pierce_2008_CountyExecutive",
     "Aspen_2009_Mayor","Berkeley_2010_D4CityCouncil","Berkeley_2010_D8CityCouncil","Oakland_2010_D6CityCouncil",
     "Pierce_2008_CityCouncil","Pierce_2008_CountyAuditor","SanFran_2007_Mayor",
     #"Minneapolis_2013_Mayor", "Minneapolis_2017_Mayor", "Minneapolis_2021_Mayor"
     ])

datafiles_nsw = np.array(["Albury"])
datafiles_usirv = np.array(["Oakland_2010_Mayor", "Pierce_2008_CountyAssessor", "Pierce_2008_CountyExecutive",
                            "SanFran_2007_Mayor"])

datafiles_nsw_ = "NSW2015/Data_NA_" + datafiles_nsw + ".txt_ballots.txt"
margins_nsw = "margins/NSW2015/Data_NA_" + datafiles_nsw + ".txt_ballots.csv"
orderings_nsw = "orderings/NSW2015/Data_NA_" + datafiles_nsw + ".txt_ballots.csv"
datafiles_usirv_ = "USIRV/" + datafiles_usirv + ".txt"
margins_usirv = "margins/USIRV/" + datafiles_usirv + ".csv"
orderings_usirv = "orderings/USIRV/" + datafiles_usirv + ".csv"

path = "/home/aek/lu91/shared/data/dirtree-elections-analysis/"  # MonARCH
# path = "/Users/aekk0001/Documents/PPR-Audits/datafiles/" # Local

datafile_names = np.concatenate((datafiles_nsw, datafiles_usirv))
datafiles = path + np.concatenate((datafiles_nsw_, datafiles_usirv_))
margins = path + np.concatenate((margins_nsw, margins_usirv))
orderings = path + np.concatenate((orderings_nsw, orderings_usirv))

runs = [("Oakland_2010_Mayor", "random", 1e-4, [range(600, 1000)]),
        ("Oakland_2010_Mayor", "random", 3e-4, [range(249, 1000)]),
        ("Oakland_2010_Mayor", "random", 1e-3, [range(260, 1000)]),
        ("Oakland_2010_Mayor", "random", 3e-3, [range(249, 1000)]),
        ("Oakland_2010_Mayor", "random", 1e-2, [range(358, 1000)]),
        ###
        ("Pierce_2008_CountyAssessor", "all_over", 1e-4, [range(942, 1000)]),
        ("Pierce_2008_CountyAssessor", "all_under", 1e-4, [range(0, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "truncate", 1e-4, [range(0, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "random", 1e-4, [range(0, 333), range(333, 666), range(666, 1000)]),
        #
        ("Pierce_2008_CountyAssessor", "all_under", 3e-4, [range(51, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "truncate", 3e-4, [range(0, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "random", 3e-4, [range(0, 333), range(333, 666), range(666, 1000)]),
        #
        ("Pierce_2008_CountyAssessor", "all_under", 1e-3, [range(48, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "truncate", 1e-3, [range(0, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "random", 1e-3, [range(0, 333), range(333, 666), range(666, 1000)]),
        #
        ("Pierce_2008_CountyAssessor", "all_under", 3e-3, [range(42, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "truncate", 3e-3, [range(0, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "random", 3e-3, [range(0, 333), range(333, 666), range(666, 1000)]),
        #
        ("Pierce_2008_CountyAssessor", "all_under", 1e-2, [range(189, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "truncate", 1e-2, [range(0, 333), range(333, 666), range(666, 1000)]),
        ("Pierce_2008_CountyAssessor", "random", 1e-2, [range(0, 333), range(333, 666), range(666, 1000)]),
        ###
        ("Pierce_2008_CountyExecutive", "all_under", 1e-4, [range(687, 1000)]),
        ("Pierce_2008_CountyExecutive", "truncate", 1e-4, [range(0, 500), range(500, 1000)]),
        ("Pierce_2008_CountyExecutive", "random", 1e-4, [range(0, 500), range(500, 1000)]),
        #
        ("Pierce_2008_CountyExecutive", "all_under", 3e-4, [range(684, 1000)]),
        ("Pierce_2008_CountyExecutive", "truncate", 3e-4, [range(0, 500), range(500, 1000)]),
        ("Pierce_2008_CountyExecutive", "random", 3e-4, [range(0, 500), range(500, 1000)]),
        #
        ("Pierce_2008_CountyExecutive", "all_under", 1e-3, [range(691, 1000)]),
        ("Pierce_2008_CountyExecutive", "truncate", 1e-3, [range(0, 500), range(500, 1000)]),
        ("Pierce_2008_CountyExecutive", "random", 1e-3, [range(0, 500), range(500, 1000)]),
        #
        ("Pierce_2008_CountyExecutive", "truncate", 3e-3, [range(221, 500), range(500, 1000)]),
        ("Pierce_2008_CountyExecutive", "random", 3e-3, [range(0, 500), range(500, 1000)]),
        #
        ("Pierce_2008_CountyExecutive", "all_under", 1e-2, [range(724, 1000)]),
        ("Pierce_2008_CountyExecutive", "truncate", 1e-2, [range(0, 500), range(500, 1000)]),
        ("Pierce_2008_CountyExecutive", "random", 1e-2, [range(0, 500), range(500, 1000)]),
        ###
        ("SanFran_2007_Mayor", "all_under", 1e-4, [range(755, 1000)]),
        ("SanFran_2007_Mayor", "truncate", 1e-4, [range(0, 500), range(500, 1000)]),
        ("SanFran_2007_Mayor", "random", 1e-4, [range(0, 500), range(500, 1000)]),
        #
        ("SanFran_2007_Mayor", "all_under", 3e-4, [range(768, 1000)]),
        ("SanFran_2007_Mayor", "truncate", 3e-4, [range(0, 500), range(500, 1000)]),
        ("SanFran_2007_Mayor", "random", 3e-4, [range(0, 500), range(500, 1000)]),
        #
        ("SanFran_2007_Mayor", "all_under", 1e-3, [range(792, 1000)]),
        ("SanFran_2007_Mayor", "truncate", 1e-3, [range(0, 500), range(500, 1000)]),
        ("SanFran_2007_Mayor", "random", 1e-3, [range(0, 500), range(500, 1000)]),
        #
        ("SanFran_2007_Mayor", "all_under", 3e-3, [range(790, 1000)]),
        ("SanFran_2007_Mayor", "truncate", 3e-3, [range(0, 500), range(500, 1000)]),
        ("SanFran_2007_Mayor", "random", 3e-3, [range(0, 500), range(500, 1000)]),
        #
        ("SanFran_2007_Mayor", "all_under", 1e-2, [range(765, 1000)]),
        ("SanFran_2007_Mayor", "truncate", 1e-2, [range(0, 500), range(500, 1000)]),
        ("SanFran_2007_Mayor", "random", 1e-2, [range(0, 500), range(500, 1000)]),
        ]

if __name__ == "__main__":
    n_runs = 10000
    erates = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    dists = ["all_over", "all_under", "truncate", "random"]
    counter = 0  # 1-642
    for i, _ in enumerate(datafiles):
        _name = str(datafile_names[i])
        runs_for_name = [run for run in runs if run[0] == _name]
        for (_, _dist, _erate, _runsets) in runs_for_name:
            for _runset in _runsets:
                counter += 1  # 1-100
                # print(counter, _name, _dist, _erate, _runset); continue
                if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue
                print("datafile, method, pop_size, margin, dist, error_rate, assorter_margin, d_value, alpha, permid"
                      "sample_size, certified, n_two_over, n_one_over, n_one_under, n_two_under")

                _ncand, _winner, _pop_size, _margin, _orderdata, _rairedata = read_election_files(datafiles[i],
                                                                                                  margins[i],
                                                                                                  orderings[i])

                main(_name, _rairedata, _pop_size, _ncand, _winner, _margin, _orderdata, _erate, _dist, n_runs,
                     runset=_runset)
        # for _erate in erates:
        #     counter += 1
        #     # print(counter); continue
        #     if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue
        #     print("datafile, method, pop_size, margin, dist, error_rate, assorter_margin, d_value, alpha, permid"
        #           "sample_size, certified, n_two_over, n_one_over, n_one_under, n_two_under")
        #
        #     _ncand, _winner, _pop_size, _margin, _orderdata, _rairedata = read_election_files(datafiles[i],
        #                                                                                       margins[i],
        #                                                                                       orderings[i])
        #     _name = str(datafile_names[i])
        #
        #     for _dist in dists:
        #         if _erate == 0.0 and _dist != "random":
        #             continue  # skip overstatement distributions for zero error rates
        #         # counter += 1
        #
        #         main(_name, _rairedata, _pop_size, _ncand, _winner, _margin, _orderdata, _erate, _dist, n_runs)
