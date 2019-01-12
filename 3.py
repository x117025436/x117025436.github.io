import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn
import pickle
import pprint
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
import os
os.chdir('C:/Users/TStra/Desktop/signal')

#get feature file
def get_feature(fname):
    pandas2ri.activate()
    robjects.r.source('feature_extract.R')
    data_read = robjects.r.processFolder(fname)
    data_read = pandas2ri.ri2py(data_read)
    return data_read

if __name__ == '__main__':
    data_list = []
    model_save = open('model.pkl', 'rb')
    model = pickle.load(model_save)
    model_save.close()

    file_name_list = os.listdir('data')             #读取声音文件
    for file_name in file_name_list:
        data = get_feature(file_name)
        data_list.append(data)
    test = pd.concat(data_list)             #获取wav特征

    pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
    print (pred)
    pre_label = np.zeros([pred.shape[0], 1])
    for i in range(pred.shape[0]):
        if pred[i] >= 0.5:
            pre_label[i] = pred[i]
        else:
            pre_label[i] = pred[i]
    num = 0
    tlen = len(pre_label)
    for i in pre_label:
        num += i
        print ('female is;'+str(num))
        print ('male is:'+str(tlen-num))
        print ((tlen-num)/tlen)
        print (num/tlen)        
