import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
a = pd.read_csv('other1.csv')
b = pd.read_csv('../input/stack-models/submission.csv')
c = pd.read_csv('../input/melanoma-efficientnet-b6-tpu-tta/submission.csv')
pd.DataFrame({'image_name': a.image_name, 'target': (a.target + c.target*0.8)/2}).to_csv('submission.csv', index=False)
