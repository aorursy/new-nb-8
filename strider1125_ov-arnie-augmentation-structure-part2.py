






import os

import sys






os.environ["ARNIEFILE"] = f"/kaggle/working/arnie.conf"

sys.path.append('/kaggle/working/draw_rna_pkg/')

sys.path.append('/kaggle/working/draw_rna_pkg/ipynb/')





import seaborn as sns

sns.set_context('poster')

sns.set_style('white')

import numpy as np

from arnie.pfunc import pfunc

from arnie.free_energy import free_energy

from arnie.bpps import bpps

from arnie.mfe import mfe

import arnie.utils as utils

from decimal import Decimal

import pandas as pd

from draw import draw_struct

from arnie.mea.mea import MEA
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

use_cols = ['id','sequence', 'structure']

all_samples = pd.concat([train[use_cols], test[use_cols]], ignore_index=True, sort=False)
all_samples = all_samples.iloc[3000:]
from collections import defaultdict

d = defaultdict(list)



for i in range(len(all_samples)):

    print(i, end=" ")

    rna_id = all_samples.id.values[i]

    sequence = all_samples.sequence.values[i]

    ground_truth_struct = all_samples.structure.values[i]

    bp_matrix = bpps(sequence)

    

    for log_gamma in range(-6,6):

        mea_mdl = MEA(bp_matrix,gamma=10**log_gamma)

        [exp_sen, exp_ppv, exp_mcc, exp_fscore] = mea_mdl.score_expected()

        [sen, ppv, mcc, fscore] = mea_mdl.score_ground_truth(ground_truth_struct)



        d[rna_id].append((exp_sen, exp_ppv, exp_mcc, exp_fscore, mea_mdl.structure))
import json

json.dump(d, open("part2.json", "w"))