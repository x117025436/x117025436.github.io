import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
import pandas as pd

os.chdir('C:/Users/TStra/Desktop/signal')

data_list = []

#获取特征文件
def get_feature(fname):
    pandas2ri.activate()
    robjects.r.source('feature_extract.R')      #利用rpy2读取R脚本
    data_read = robjects.r.processFolder(fname) #得到数据文件
    data_read = pandas2ri.ri2py(data_read)      #转化为python可以使用的数据
    return data_read

if __name__ == '__main__':
    file_name_list = os.listdir('data')         #存放.wav格式声音的文件夹
    for file_name in file_name_list:
        data = get_feature(file_name)
        data_list.append(data)
    result = pd.concat(data_list)           
    result['label'] = 'male'
    result.to_csv("male.csv", index=False)
    #result['label'] = 'female'
    #result.to_csv("female.csv", index=False)
