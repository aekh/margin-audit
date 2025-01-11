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

from raire_audit import read_election_files

def main(name="", error_rate=0.03, margin=0.03, size=50_000, orderdata=None):
    # Create CVR data file
    n_mismatch = int(round(size * error_rate))

    #assert n_mismatch / size == error_rate, "Given error pct not attainable"  # Ensure there is no rounding error
    if margin < 1: # margin is given as a percentage
        ballot_margin = int(size * margin)
        assert ballot_margin / size == margin, "Given margin pct not attainable"  # Ensure there is no rounding error
    else:
        ballot_margin = margin
        margin = ballot_margin / size
        # assert int(round(margin * size)) == ballot_margin, "Given margin pct not attainable"  # Ensure there is no rounding error

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
    pvalue_histories_array, overstatements_history = calc_pvalues_all_orderings(contests, cvr_input, orderdata=orderdata, n_mismatch=n_mismatch, size=size)
    # pvalue_histories_array, overstatements_history = calc_pvalues_all_orderings(contests, cvr_input)
    for pvalue_history in pvalue_histories_array:
        below_5pct = pvalue_history <= 0.05
        certified_5pct = True
        where_5pct = np.argmax(below_5pct) + 1
        if sum(below_5pct) == 0 or pvalue_history[where_5pct-1] == 0.0:
            certified_5pct = False
            where_5pct = len(pvalue_history)
        print(f"{name}, {margin}, {size}, {error_rate}, {min_margin}, {where_5pct}, {certified_5pct}")


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
def calc_pvalues_single_ordering(contests, cvr_input, rng, ordering_index, orderdata, n_mismatch, size):
    #print("Working on ordering {}".format(ordering_index))

    if orderdata is None:
        pass  # todo: implement random ordering
        cvr_input_shuffled = deepcopy(cvr_input)
        rng.shuffle(cvr_input_shuffled)
    else:
        error_ids = orderdata[ordering_index-1][:n_mismatch]

        cvr_input_shuffled = []
        for i in range(size - n_mismatch):
            cvr_input_shuffled.append({"id": "match"+str(i+1), "votes": {"1": {"match": 1, "mismatch": 0}}})
        for idx in sorted(error_ids):
            cvr_input_shuffled.insert(idx, {"id": "mismatch"+str(idx+1), "votes": {"1": {"match": 0, "mismatch": 1}}})

    # Import shuffled CVRs.
    shuffled_ballots = CVR.from_dict(cvr_input_shuffled)

    # Find measured risks for all assertions.
    Assertion.set_p_values(contests, shuffled_ballots, None)

    # Extract all of the p-value histories and combine them.
    pvalues = merge_pvalues(contests['1'].assertions)

    return pvalues


# Calculate p-values for a set of orderings.
def calc_pvalues_all_orderings(contests, cvr_input, n_orderings=10, orderdata=None, n_mismatch=0, size=0):
    pvalue_list = []
    rng = np.random.default_rng(seed=2024_10_29)
    for o in range(n_orderings):
        pvalue_list.append(calc_pvalues_single_ordering(contests, cvr_input, rng, o, orderdata, n_mismatch, size))
    # pvalue_array = np.stack(pvalue_list, axis=-1)
    return pvalue_list, []


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
     "Minneapolis_2013_Mayor", "Minneapolis_2017_Mayor", "Minneapolis_2021_Mayor"])

datafiles_nsw_ = "NSW2015/Data_NA_" + datafiles_nsw + ".txt_ballots.txt"
margins_nsw = "margins/NSW2015/Data_NA_" + datafiles_nsw + ".txt_ballots.csv"
orderings_nsw = "orderings/NSW2015/Data_NA_" + datafiles_nsw + ".txt_ballots.csv"
datafiles_usirv_ = "USIRV/" + datafiles_usirv + ".txt"
margins_usirv = "margins/USIRV/" + datafiles_usirv + ".csv"
orderings_usirv = "orderings/USIRV/" + datafiles_usirv + ".csv"

# path = "/home/aek/lu91/shared/data/dirtree-elections-analysis/"  # MonARCH
path = "/Users/aekk0001/Documents/PPR-Audits/datafiles/" # Local

datafile_names = np.concatenate((datafiles_nsw, datafiles_usirv))
datafiles = path + np.concatenate((datafiles_nsw_, datafiles_usirv_))
margins = path + np.concatenate((margins_nsw, margins_usirv))
orderings = path + np.concatenate((orderings_nsw, orderings_usirv))


if __name__ == "__main__":
    counter = 0  # array size on SLURM cluster == 1-13
    erates = [0.0000, 0.0001, 0.0005, 0.0010, 0.0050]
    for i, _ in enumerate(datafiles):
        counter += 1  # 1-110
        # print(counter); continue
        # if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue  # parallelise on SLURM cluster

        ncand, winner, nballots, margin, orderdata, rairedata = read_election_files(datafiles[i], margins[i], orderings[i])

        print("datafile, margin, pop_size, error_rate, assertion_margin, sample_size_5pct, certified_5pct, 2-overstatements")

        datafile_name = str(datafile_names[i])
        # datafile = datafiles[i]
        marginfile = margins[i]
        # size = 0
        # margin = -1
        # with open(datafile) as f:
        #     for line in f:
        #         if ":" in line:
        #             size += int(line.split(":")[1])
        with open(marginfile) as f:
            f.readline()
            for line in f:
                margin = max(margin, int(line.split(",")[2]))
        # # print(f"{datafile_name=}, {margin=}, {size=}, {erate=}")
        for erate in erates:
            main(name=datafile_name, error_rate=erate, margin=margin, size=nballots, orderdata=orderdata)
    # margins = [0.0001, 0.0005, 0.0010, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500,
    #            0.0600, 0.0700, 0.0800, 0.0900, 0.1000, 0.1250, 0.1500]
    # erates = [0.0000, 0.0001, 0.0005, 0.0010, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200, 0.0250, 0.0300, 0.0400,
    #           0.0500]
    # for margin in margins:
    #     counter += 1
    #     # if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue  # parallelise on SLURM cluster
    #     print("pop_size, error_rate, margin, assertion_margin, sample_size_10pct, sample_size_5pct, sample_size_1pct, "
    #       "certified_10pct, certified_5pct, certified_1pct")
    #     for erate in erates:
    #         main(error_rate=erate, margin=margin)
