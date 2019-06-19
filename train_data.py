import pandas as pd
import threading
import numpy as np
import chardet
import datetime
from sklearn.preprocessing  import OneHotEncoder,Imputer
from scipy import sparse
def train_test_data():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df_train = df_train.loc[df_train["repay_date"]!="\\N",:]
    df_train = df_train.loc[df_train["repay_amt"]!="\\N",:]
    return df_train,df_test
def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测文件编码类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']
def user_taglist():
    user_taglist = pd.read_csv("user_taglist.csv")
    return user_taglist
def listing_info():
    listing_info = pd.read_csv("listing_info.csv")
    return listing_info
def user_behavior_logs():
    user_behavior_logs = pd.read_csv("user_behavior_logs.csv")
    return user_behavior_logs
def user_repay_logs():
    user_repay_logs = pd.read_csv("user_repay_logs.csv")
    return user_repay_logs
def user_info():
    user_info = pd.read_csv("user_info.csv",encoding="utf-8")
    user_info = user_info.loc[user_info["id_city"] != "\\N", :]
    return user_info
def One_Hot(df_total,df_train,df_test,columns,num_col):
    enc = OneHotEncoder(categories='auto')
    df_train_ = df_train.loc[:,num_col].values
    df_test_ = df_test.loc[:,num_col].values
    for column in columns:
        if 'id' in column:
            continue
        else:
            enc.fit(df_total[column].values.reshape(-1,1))
            df_train_ = sparse.hstack((df_train_, enc.transform(df_train[column].values.reshape(-1, 1))),'csr')
            df_test_ = sparse.hstack((df_test_, enc.transform(df_test[column].values.reshape(-1, 1))), 'csr')
    return df_train_, df_test_
def label_encoder(df,columns):# 去除id
    for column in columns:
        if 'id' in column:
            continue
        else:
            df.loc[:,column] = df[column].map(dict(zip(df.loc[:,column].unique(),range(0,df.loc[:,column].nunique()))))
    return df
def split_bins(df,df1,target_col,cols,map_col,n,bool):
    df.loc[:,target_col] = df.loc[:,cols].values
    df = pd.get_dummies(df, columns=[target_col])
    li = []
    for i in range(1, n+1):
        str1 = target_col+"_" + str(i)
        li.append(str1)
    mean_columns = []
    del_col = list(df.columns)
    del_col.remove('happiness')
    for f1 in del_col:
        cate_rate = df.loc[:,f1].value_counts(normalize=True, dropna=False).values[0]
        if cate_rate <=0.8:
            for f2 in li:
                col_name = map_col+'_to_' + f1 + "_" + f2 + '_mean'
                mean_columns.append(col_name)
                if f2 not in df.columns:
                    li.remove(f2)
                    continue
                else:
                    order_label = df.groupby([f1])[f2].mean()
                    # print(order_label)
                    df.loc[:,col_name] = df.loc[:,map_col].map(order_label)
                    df1.loc[:,col_name] = df1.loc[:,map_col].map(order_label)
                    miss_rate = df.loc[:,col_name].isnull().sum() * 100 / df.loc[:,col_name].shape[0]
                    if miss_rate > 0:
                        df = df.drop([col_name], axis=1)
                        df1 = df1.drop([col_name], axis=1)
                        mean_columns.remove(col_name)
    if bool:
        df = df.drop(li,axis=1)
    return df,df1
def run(df_test_user_taglist_listing_info_user_info):
    print("子线程开始了。。。。。。。。。。。。")
    for index in list(df_test_user_taglist_listing_info_user_info.index):
        print("子线程执行。。。。。。。。。。。。")
        year, month, day = str(df_test_user_taglist_listing_info_user_info.loc[index, 'auditing_date']).split( sep="-")
        year_1, month_1, day_1 = str(df_test_user_taglist_listing_info_user_info.loc[index, 'due_date']).split(sep="-")
        starttime = datetime.datetime(int(year), int(month), int(day))
        endtime = datetime.datetime(int(year_1), int(month_1), int(day_1))
        df_test_user_taglist_listing_info_user_info.loc[index, 'due_date'] = endtime-starttime
    df_test_user_taglist_listing_info_user_info.to_csv("test_data.csv")
if __name__ == "__main__":
    df_train,df_test = train_test_data()
    #与用户画像标签链接
    user_taglist = user_taglist()
    user_taglist = user_taglist.drop(['insertdate'],axis=1)
    df_train_user_taglist = pd.merge(df_train,user_taglist,on='user_id',how='left')
    df_test_user_taglist = pd.merge(df_test, user_taglist, on='user_id', how='left')
    print(df_train_user_taglist.shape)
    #与标的属性表链接
    listing_info = listing_info()
    listing_info = listing_info.drop(["auditing_date"],axis =1)
    df_train_user_taglist_listing_info = pd.merge(df_train_user_taglist,listing_info,on = ['user_id','listing_id'],how = 'left')
    df_test_user_taglist_listing_info = pd.merge(df_test_user_taglist, listing_info, on=['user_id', 'listing_id'],how='left')
    #与借款用户基础信息表链接
    user_info = user_info()
    user_info = user_info.drop(['reg_mon','cell_province',"insertdate",'id_province','id_city'],axis=1)
    df_train_user_taglist_listing_info_user_info = pd.merge(df_train_user_taglist_listing_info,user_info,on ='user_id',how = 'left')
    df_test_user_taglist_listing_info_user_info = pd.merge(df_test_user_taglist_listing_info, user_info, on='user_id', how='left')
    print(df_train_user_taglist_listing_info_user_info.shape)
    #属性日期转换
    print(df_train_user_taglist_listing_info_user_info.columns)
    print('这是主线程：', threading.current_thread().name)
    #创建子线程
    thread_test = threading.Thread(target= run,kwargs={'df_test_user_taglist_listing_info_user_info':df_test_user_taglist_listing_info_user_info})
    thread_test.start()
    #x训练集
    for index in list(df_train_user_taglist_listing_info_user_info.index):
        print("主线程执行。。。。。。。。。。。。")
        year, month, day= str(df_train_user_taglist_listing_info_user_info.loc[index, 'auditing_date']).split( sep="-")
        year_1, month_1, day_1 = str(df_train_user_taglist_listing_info_user_info.loc[index, 'due_date']).split(sep="-")
        year_2, month_2, day_2 = str(df_train_user_taglist_listing_info_user_info.loc[index, 'repay_date']).split(sep="-")
        starttime = datetime.datetime(int(year),int(month),int(day))
        endtime = datetime.datetime(int(year_1), int(month_1), int(day_1))
        endtime2 = datetime.datetime(int(year_2), int(month_2), int(day_2))
        df_train_user_taglist_listing_info_user_info.loc[index, 'due_date'] = endtime-starttime
        df_train_user_taglist_listing_info_user_info.loc[index, 'repay_date'] = endtime2 - starttime
    #保存下训练和测试数据。下次运行就可以不执行前面的步骤
    df_train_user_taglist_listing_info_user_info.to_csv("train_data.csv")
    #主线程等待子线程结束
    thread_test.join()