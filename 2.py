#-*- coding:utf-8 _*-  
import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import pickle
import pprint

def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', sklearn.metrics.log_loss(labels, preds)

input_data = pd.read_csv('C:/Users/TStra/Desktop/signal/voice.csv') #pandas 读取csv文件
input_data = input_data.sample(frac=1) #利用pandas抽样 frac=1 即比例为1
gender = {'male' : 0, 'female' : 1} #性别判断
input_data['label'] = input_data['label'].map(gender) #map函数
cols = [c for c in input_data.columns if c not in ['label']]
print (cols)
train = input_data.iloc[0 :3300]
test = input_data.iloc[3300 : ]
test_label = test['label']
test_label = np.array(test_label).reshape([-1 , 1])
del(test['label'])

fold = 1
for i in range(fold):
    params = {
        'eta': 0.01, #use 0.002
        'max_depth': 5,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'lambda':0.1,
        'gamma':0.1,
        'seed': i,
        'silent': True
    }
    x1 = train[cols][0:3000]
    x2 = train[cols][3000:]
    y1 = train['label'][0:3000]
    y2 = train['label'][3000 : ]
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1500,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

pred /= fold
pre_label = np.zeros([pred.shape[0], 1])
for i in range(pred.shape[0]):
    if pred[i] >= 0.5:
        pre_label[i] = 1
    else:
        pre_label[i] = 0

acc = np.mean(np.equal(pre_label, test_label).astype(np.float))
print("the test acc is:", acc)

model_save = open('model.pkl', 'wb')    #保存模型
pickle.dump(model, model_save)
model_save.close()