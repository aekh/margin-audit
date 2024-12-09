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

nalice = 55
nbob = 45
ballot_margin = (nalice - nbob)/2
margin = ballot_margin / (nalice + nbob)

print(f"Number of ballots total: {nalice + nbob}, votes for Alice: {nalice}, votes for Bob: {nbob}, ballot-margin: {ballot_margin}, margin: {margin}")

#######################
# ballot-margin audit #
#######################

# Create CVR data file
cvr_input = []
for i in range(nalice + nbob):
    cvr_input.append({"id": "match" + str(i + 1), "votes": {"1": {"match": 1, "mismatch": 0}}})

cvr_list = CVR.from_dict(cvr_input)
max_cards = len(cvr_list)

audit = Audit.from_dict({
    'strata': {'stratum1': {'use_style': True,
                            'replacement': False}
               }
})

contest_dict = {'1': {'name': '1',
                      'risk_limit': 0.05,
                      'cards': max_cards,
                      'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                      'n_winners': 1,
                      'share_to_win': 1 - margin,
                      'candidates': ['match', 'mismatch'],
                      'winner': ['match'],
                      'audit_type': Audit.AUDIT_TYPE.POLLING,
                      'test': NonnegMean.alpha_mart,
                      'estim': NonnegMean.shrink_trunc
                      }
                }

contests = Contest.from_dict_of_dicts(contest_dict)

# Construct the dict of dicts of assertions for each contest.
Assertion.make_all_assertions(contests)
audit.check_audit_parameters(contests)

# Calculate margins for each assertion.
min_margin = Assertion.set_all_margins_from_cvrs(audit, contests, cvr_list)

print(f"Ballot-Margin Audit: Assertion margin: {min_margin}")

# calculate p-values

rng = np.random.default_rng(seed=2024_12_09)

# Shuffle ballots.
cvr_input_shuffled = deepcopy(cvr_input)
rng.shuffle(cvr_input_shuffled)
shuffled_ballots = CVR.from_dict(cvr_input_shuffled)

# Find measured risks for all assertions.
Assertion.set_p_values(contests, shuffled_ballots, None)

pvalues = np.array(list(contests['1'].assertions.values())[0].p_history)
samplesize = np.argmax(pvalues <= 0.05) + 1

print(f"Ballot-Margin Audit: Sample size: {samplesize}")

#########################
# card-comparison audit #
#########################

# Create CVR data file
cvr_input = []
for i in range(nalice):
    cvr_input.append({"id": "alice"+str(i+1), "votes": {"1": {"alice": 1, "bob": 0}}})
for i in range(nbob):
    cvr_input.append({"id": "bob"+str(i+1), "votes": {"1": {"alice": 0, "bob": 1}}})

cvr_list = CVR.from_dict(cvr_input)
max_cards = len(cvr_list)

audit = Audit.from_dict({
    'strata': {'stratum1': {'use_style': True,
                            'replacement': False}
               }
})

contest_dict = {'1': {'name': '1',
                      'risk_limit': 0.05,
                      'cards': max_cards,
                      'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                      'n_winners': 1,
                      'share_to_win': 1/2,
                      'candidates': ['alice', 'bob'],
                      'winner': ['alice'],
                      'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                      'test': NonnegMean.alpha_mart,
                      'estim': NonnegMean.shrink_trunc
                      }
                }

contests = Contest.from_dict_of_dicts(contest_dict)

# Construct the dict of dicts of assertions for each contest.
Assertion.make_all_assertions(contests)
audit.check_audit_parameters(contests)

# Calculate margins for each assertion.
min_margin = Assertion.set_all_margins_from_cvrs(audit, contests, cvr_list)

print(f"Card Comparison Audit: Assertion margin: {min_margin}")

# calculate p-values

rng = np.random.default_rng(seed=2024_12_09)

# Shuffle ballots.
cvr_input_shuffled = deepcopy(cvr_input)
rng.shuffle(cvr_input_shuffled)
shuffled_cvrs = CVR.from_dict(cvr_input_shuffled)
shuffled_mvrs = CVR.from_dict(cvr_input_shuffled)

# Find measured risks for all assertions.
Assertion.set_p_values(contests, cvr_sample=shuffled_cvrs, mvr_sample=shuffled_mvrs, use_all=True)

pvalues = np.array(list(contests['1'].assertions.values())[0].p_history)
samplesize = np.argmax(pvalues <= 0.05) + 1

print(f"Card Comparison Audit: Sample size: {samplesize}")
