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


# main(name=datafile_name, data=rairedata, ncand=ncand, size=nballots, orderdata=orderdata, error_rate=0.0,
#      margin=margin)

def main(name="", data=None, ncand=None, winner=None, size=None, orderdata=None, error_rate=0.00, margin=0.03):

    if error_rate==0.0:
        n_orderings = 1
        n_errors = 0
    else:
        n_orderings = 1000
        n_errors = int(round(size * error_rate))

    contests, cvrs = load_contests_from_raire_raw(data)
    contest = contests[0]

    cvr_input = [{"id": c, "votes": {'1': {i: cvrs[c]['1'][i]+1 for i in cvrs[c]['1'].keys()}}} for c in cvrs.keys()]

    assertions = compute_raire_assertions(contest, cvrs, str(winner), cp_estimate, log=False)

    assertions = [a.to_json() for a in assertions]

    #cvr_list, num_cvrs_read = CVR.from_raire(cvr_input)
    cvr_list = CVR.from_dict(cvr_input)

    max_cards = len(cvr_list)
    # print(f'After merging, there are CVRs for {max_cards} cards')

    # Set specifications for the audit.
    audit = Audit.from_dict({
        'strata' : {'stratum1' : {'use_style'   : True,
                                  'replacement' : False}
            }
        })

    contest_dict = {'1': {'name': '1',
                          'risk_limit': 0.05,
                          'cards': max_cards,
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

    # Construct the dict of dicts of assertions for each contest.
    Assertion.make_all_assertions(contests)
    audit.check_audit_parameters(contests)

    # Calculate margins for each assertion.
    min_margin = Assertion.set_all_margins_from_cvrs(audit, contests, cvr_list)
    # Get min margin assertion
    hardest_assertion_name, hardest_assertion = min(contests['1'].assertions.items(), key=lambda x: x[1].margin)

    # Calculate all of the p-values.
    pvalue_histories_array, overstatements_history = calc_pvalues_all_orderings(contests, cvr_input, hardest_assertion, orderdata=orderdata, n_orderings=n_orderings, n_errors=n_errors)
    for i, pvalue_history in enumerate(pvalue_histories_array):
        below_5pct = pvalue_history <= 0.05
        overstatements = np.array(overstatements_history[i])
        overstatement_type, counts = np.unique(overstatements[~below_5pct], return_counts=True)
        value_counts = dict(zip([int(x) for x in overstatement_type], [int(x) for x in counts]))
        value_counts.setdefault(-2, 0)
        value_counts.setdefault(-1, 0)
        value_counts.setdefault(0, 0)
        value_counts.setdefault(1, 0)
        value_counts.setdefault(2, 0)
        certified_5pct = True
        where_5pct = np.argmax(below_5pct) + 1
        if sum(below_5pct) == 0 or pvalue_history[where_5pct-1] == 0.0:
            certified_5pct = False
            where_5pct = len(pvalue_history)
        print(f"{name}, {margin}, {size}, {error_rate}, {min_margin}, {where_5pct}, {certified_5pct}, "
              f"{value_counts[-2]}, {value_counts[-1]}, {value_counts[1]}, {value_counts[2]}")


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
def calc_pvalues_single_ordering(contests, cvr_input, hardest_assertion, orderdata, ordering_index, n_errors=0):
    # get ordering for sampling and ordering for applying errors
    ordering = orderdata[ordering_index]
    error_ids = orderdata[ordering_index-1][:n_errors]

    # Shuffle ballots according to the given ordering.
    cvr_input_shuffled = shuffle(cvr_input, ordering)
    mvr_input_shuffled = deepcopy(cvr_input_shuffled)

    # Add errors to MVRs
    seed = ordering[0]
    rng = np.random.default_rng(seed)
    ncand = len(contests['1'].candidates)
    for i in error_ids:
        votes = mvr_input_shuffled[i-1]['votes']['1']
        nprefs = len(votes)
        if nprefs == 0:
            add = rng.integers(0, ncand)
            new_votes = {str(add): 1}
        elif nprefs == 1:
            new_votes = {}
        else:
            cut_at = rng.integers(0, nprefs-1)
            new_votes = {cand: votes[cand] for cand in votes.keys() if votes[cand] <= cut_at}  # TODO This is wrong
        # print(f'Error: {i}, {votes} -> {new_votes}')
        mvr_input_shuffled[i-1]['votes']['1'] = new_votes

    # Import shuffled CVRs.
    shuffled_cvrs = CVR.from_dict(cvr_input_shuffled)
    shuffled_mvrs = CVR.from_dict(mvr_input_shuffled)

    # Find measured risks for all assertions.
    Assertion.set_p_values(contests, cvr_sample=shuffled_cvrs, mvr_sample=shuffled_mvrs, use_all=True)  # add cvrs and mvrs

    # Extract all of the p-value histories and combine them.
    pvalues = merge_pvalues(contests['1'].assertions)
    overstatements = np.array([hardest_assertion.assorter.overstatement(shuffled_cvrs[i], shuffled_mvrs[i])*2 for i in range(len(shuffled_cvrs))])

    return pvalues, overstatements


# Calculate p-values for a set of orderings.
def calc_pvalues_all_orderings(contests, cvr_input, hardest_assertion, orderdata=None, n_orderings=1000, n_errors=0):
    pvalue_list = []
    overstatement_list = []
    for o in range(n_orderings):
        pvalues, overstatements = calc_pvalues_single_ordering(contests, cvr_input, hardest_assertion, orderdata, o, n_errors=n_errors)
        pvalue_list.append(pvalues)
        overstatement_list.append(overstatements)
    return pvalue_list, overstatement_list


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


if __name__ == "__main__":
    counter = 0  # array size on SLURM cluster == 1-13
    erates = [0.0000, 0.0001, 0.0005, 0.0010, 0.0050]
    for i, _ in enumerate(datafiles):
        counter += 1  # 1-107
        # print(counter); continue
        if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue  # parallelise on SLURM cluster

        ncand, winner, nballots, margin, orderdata, rairedata = read_election_files(datafiles[i], margins[i], orderings[i])

        print("datafile, margin, pop_size, error_rate, assertion_margin, sample_size_5pct, certified_5pct, "
              "2-overstatements, 1-overstatements, 1-understatements, 2-understatements")

        datafile_name = str(datafile_names[i])
        for erate in erates:
            main(name=datafile_name, data=rairedata, ncand=ncand, winner=winner, size=nballots, orderdata=orderdata,
                 error_rate=erate, margin=margin)
