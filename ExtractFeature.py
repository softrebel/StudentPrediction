import pandas as pd
import numpy as np

data = pd.read_csv('Activity.csv')
d=data[['raisedhands','VisITedResources','AnnouncementsView','Discussion','ParentschoolSatisfaction','StudentAbsenceDays','Class']]
out=d.loc[d['Class'] != 'M']
out.to_csv('ArticleDataset.csv',index=False)