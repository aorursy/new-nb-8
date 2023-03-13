# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# only contain the display_id and ad_id. 

cli_test_demo = pd.read_csv('../input/clicks_test.csv', nrows=10)

cli_test_demo
# the clicks_train file contain more information like the clicked item

cli_train_demo = pd.read_csv('../input/clicks_train.csv', nrows=10)

cli_train_demo
# note here is the confidence_level. By this way we can decrease the vector length by projection the document id to category id



do_ca = pd.read_csv('../input/documents_categories.csv', nrows=10)

do_ca
#Don't quite understand about this file

#documents_entities.csv give the confidence that the given entity was referred to in the document.

#an entity_id can represent a person, organization, or location

#Maybe useless than the categories files

do_en = pd.read_csv('../input/documents_entities.csv', nrows=10)

do_en

#Could it decrease the length of the document vector? I don't think so.

#which will make more important for a document, its content or who publish it? its content

do_me = pd.read_csv('../input/documents_meta.csv', nrows=10)

do_me

#We may ignore this file

#Or we may include the source_id
#the same format as the categories and entities file

do_to = pd.read_csv('../input/documents_topics.csv', nrows=10)

do_to
ev = pd.read_csv('../input/events.csv', nrows=10)

ev
pa = pd.read_csv('../input/page_views_sample.csv', nrows=10)

pa
pr_co = pd.read_csv('../input/promoted_content.csv', nrows=10)

pr_co
import pandas as pd

import numpy as np 
reg = 10 # trying anokas idea of regularization

eval = True



train = pd.read_csv("../input/clicks_train.csv")



if eval:

	ids = train.display_id.unique()

	ids = np.random.choice(ids, size=len(ids)//10, replace=False)



	valid = train[train.display_id.isin(ids)]

	train = train[~train.display_id.isin(ids)]

	

	print (valid.shape, train.shape)



cnt = train[train.clicked==1].ad_id.value_counts() #total count of clicked ad_id

cntall = train.ad_id.value_counts() # total count of all the ad_id, use to normalize
del train



def get_prob(k):

    if k not in cnt:

        return 0

    return cnt[k]/(float(cntall[k]) + reg)



def srt(x):

    ad_ids = map(int, x.split())

    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)

    return " ".join(map(str,ad_ids)) 
if eval:

	from ml_metrics import mapk

	

	y = valid[valid.clicked==1].ad_id.values

	y = [[_] for _ in y]

	p = valid.groupby('display_id').ad_id.apply(list)

	p = [sorted(x, key=get_prob, reverse=True) for x in p]

	

	print (mapk(y, p, k=12))

else:

	subm = pd.read_csv("../input/sample_submission.csv") 

	subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))

	subm.to_csv("subm_reg_1.csv", index=False)
subm = pd.read_csv("../input/sample_submission.csv", nrows=10)
subm
subm.ad_id.apply(lambda x: srt(x))
re = subm.ad_id.apply(lambda x: [get_prob(i) for i in map(int, x.split())])
re[0]