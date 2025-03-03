import pandas as pd
import pickle 
from sklearn.ensemble import RandomForestClassifier

penguin = pd.read_csv('datas/penguin_data.csv')

df = penguin.copy()
target = 'sex'
encode = ['sex','island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]

target_map = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_map[val]

df['species'] = df['species'].apply(target_encode)

X = df.drop('species',axis=1)
y = df['species']

clf = RandomForestClassifier()
clf.fit(X,y)

pickle.dump(clf, open('model.pkl','wb'))