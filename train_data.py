import pandas as pd
import numpy as np
import chardet
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