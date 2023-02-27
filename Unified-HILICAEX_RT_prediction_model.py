import numpy as np
import pandas as pd
import pickle
with open('Unified-HILICAEX_RT_prediction_model.pickle', mode='rb') as fp:
    clf = pickle.load(fp)

pd_1 = pd.read_csv('InputData.csv',encoding="shift-jis",header=0)
pd_2 = pd_1.values.tolist()
HeaderNumber = len(pd_1.columns)
npArray = np.array(pd_2)
MDs = npArray[:, 0:HeaderNumber]
predict = clf.predict(MDs)
df = pd.DataFrame(predict)
df.to_csv("Predicted_result.csv")
print('FINISH')
