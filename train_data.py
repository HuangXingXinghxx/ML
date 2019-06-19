import pandas as pd
import threading
import numpy as np
import chardet
import datetime
from sklearn.preprocessing  import OneHotEncoder
from scipy import sparse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
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
"""
    onehot编码，返回稀疏矩阵
    df_total为df_train和df_test的合并，考虑每一列数据的值的所有分布，而不是只考虑训练数据
    num_col为不需要onehot的列
    columns需要进行onehot的列
"""
def One_Hot(df_total,df_train,df_test,columns,num_col):
    enc = OneHotEncoder(categories='auto')
    df_train_ = df_train.loc[:,num_col].values
    df_test_ = df_test.loc [:,num_col].values
    for column in columns:
        if 'user_id' in column:#表示用户id不进行
            continue
        else:
            enc.fit(df_total[column].values.reshape(-1,1))
            df_train_ = sparse.hstack((df_train_, enc.transform(df_train[column].values.reshape(-1, 1))),'csr')
            df_test_ = sparse.hstack((df_test_, enc.transform(df_test[column].values.reshape(-1, 1))), 'csr')
    return df_train_, df_test_
"""
    值编码，即把离散型数据变成连续型离散数据，为onehot函数做准备
"""
def label_encoder(df,columns):# 去除id
    for column in columns:
        if 'user_id' in column:#表示用户id不进行
            continue
        else:
            df.loc[:,column] = df[column].map(dict(zip(df.loc[:,column].unique(),range(0,df.loc[:,column].nunique()))))
    return df
"""
    分箱，即回归目标的范围分箱多个箱子（子范围），把各类特征对应到每个箱子中，减少连续型数值的影响和异常数据的影响
"""
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

"""
  线程的运行函数
"""
def run(df_test_user_taglist_listing_info_user_info):
    print("子线程开始了。。。。。。。。。。。。")
    df_test_user_taglist_listing_info_user_info['auditing_date'] = pd.to_datetime(df_test_user_taglist_listing_info_user_info.auditing_date)
    df_test_user_taglist_listing_info_user_info['due_date'] = pd.to_datetime(df_test_user_taglist_listing_info_user_info.due_date)
    df_test_user_taglist_listing_info_user_info['due_date'] = df_test_user_taglist_listing_info_user_info['due_date'] - df_test_user_taglist_listing_info_user_info['auditing_date']
    f = lambda x:str(x)[0:2]
    df_test_user_taglist_listing_info_user_info['due_date'] = df_test_user_taglist_listing_info_user_info['due_date'].map(f)
    df_test_user_taglist_listing_info_user_info.to_csv("test_data.csv")
"""
获得所有用户的总标签列表，和每个用户的标签集合
taglist：列表类型
user_taglist：字典类型
"""
def get_taglist(df_train):
    taglist = []
    user_taglist = {}
    for (name, groupuser) in df_train[['user_id', 'taglist']].groupby(df_train['user_id'].values):
        tag_list = []
        for index in groupuser.index:
            subtag_list = str(groupuser.loc[index,'taglist']).split(sep='|')
            if len(subtag_list)<=0:
                continue
            else:
                for item in subtag_list:
                    tag_list.append(item)
                    taglist.append(item)
        user_taglist[name] = tag_list
        taglist = list(np.unique(taglist))
        tag_list.clear()
    return taglist,user_taglist

"""  
     把taglist中得每个用户得多个标签进行onehot
     把返回的df与原来的df_train拼接
     taglist：所有用户的tag总和
     user_taglist：指训练集或者测试集中的用户所对应的tag集合，为字典类型
     user_id:指训练集或者测试集中的用户集合
     length:为训练集或者测试集大小，即索引大小
"""
def taglist_onehot(taglist,user_taglist,user_id,length):
    list_index  = [x for x in range(length)]
    list_columns = [x for x in range(len(taglist))]
    df = pd.DataFrame([],index = list_index,columns =list_columns )
    df.loc[:,'user_id']= user_id.loc[:,'user_id']
    for index in list(df.index):
        #取出每个用户对应得标签集合
        tag_list = user_taglist[df.loc[index,'user_id']]
        for item in tag_list:
            position = taglist.index(item)
            df.loc[index,position] =1  #出现的tag填充为1
    df = df.fillna(0)#没出现位置填充为0
    return df
"""
    处理性别
"""
def gender(df):
    gender_dict={'男':0,'女':1}
    df["gender"] = df['gender'].map(gender_dict)
    return df
"""
    模型训练函数
"""
def train():
    df_train =  pd.read_csv("train_data.csv")
    df_test = pd.read_csv("test_data.csv")
    #保存成交日期，为后面数据回复
    df_train_auditing_date = df_train['auditing_date'].values
    df_test_auditing_date = df_test['auditing_date'].values
    df_train = df_train.drop(["auditing_date"],axis =1)
    df_test = df_test.drop(["auditing_date"],axis =1)

    # print("第一步。。。。。。。。。。。。。。")
    # #删除'repay_date'和repay_amt为空的值
    # df_train = df_train.loc[df_train['repay_date'].notnull(),:]
    # df_train = df_train.loc[df_train['repay_amt'].notnull(), :]
    #
    # repay_date_dict = {}
    # repay_amt_dict = {}
    # ##根据用户id进行分组，所以每组有一个名字name=user_id
    # for (name,groupuser) in df_train[['user_id','repay_amt','repay_date']].groupby(df_train['user_id'].values):
    #     repay_amt_mean = groupuser['repay_amt'].mean()
    #     repay_date_mean = groupuser['repay_date'].mean()
    #     #存入字典
    #     repay_date_dict[name] = repay_date_mean
    #     repay_amt_dict[name] = repay_amt_mean
    # #repay_amt和repay_date转为差距值
    # for index in list(df_train.index):
    #     df_train.loc[index,'repay_amt'] = float(df_train.loc[index,'repay_amt']) - repay_amt_dict[df_train.loc[index,'user_id']]
    #     df_train.loc[index, 'repay_date'] = float(df_train.loc[index, 'repay_date']) - repay_date_dict[df_train.loc[index, 'user_id']]


    print("第二步。。。。。。。。。。。。。。")
    #处理用户画像列；'taglist'

    #得到训练集和测试集的用户的tag和各自的总和tag
    taglist_train, user_taglist_train = get_taglist(df_train)
    taglist_test, user_taglist_test = get_taglist(df_test)
    #得到总和的tag，即tag总和
    taglist = list(set(taglist_train + taglist_test))
    #为taglist做准备
    length_train = len(df_train)
    length_test = len(df_test)
    user_id_train = df_train[['user_id']]
    user_id_test = df_test[['user_id']]
    #对taglist进行onehot，这个列不同于其他列，其他列可以用One_Hot函数
    train_df = taglist_onehot(taglist,user_taglist_train,user_id_train,length_train)
    test_df = taglist_onehot(taglist,user_taglist_test,user_id_test,length_test)
    #删除多余的user_id列
    train_df = train_df.drop(['user_id'],axis=1)
    test_df = test_df.drop(['user_id'],axis=1)
    #删除原来的taglist
    df_train = df_train.drop(['taglist'],axis=1)
    df_test = df_test.drop(['taglist'],axis=1)
    #合并
    df_train  = pd.concat([df_train,train_df],axis=0)
    df_test = pd.concat([df_test,test_df],axis=0)

    #发现其他列没必要进行onhot饿，进行数据归一化，使得每列数据的对目标的影响是均匀的，平等看待每一列数据
    #处理性别,也可以进行onehot,个人觉得二值数据影响不大
    df_train = gender(df_train)
    df_test = gender(df_test)
    #可进行split_bins,

    print("第三步。。。。。。。。。。。。。。")
    # 目标1和2
    label_repay_date = df_train['repay_date'].values
    label_repay_amt = df_train['repay_amt'].values
    df_train = df_train.drop(['repay_date','repay_amt'],axis=1)

    #保存训练集和测试集的行列索引
    train_columns = list(df_train.columns)
    test_columns = list(df_test.columns)
    train_index = list(df_train.index)
    test_index = list(df_test.index)

    #保存测试集的listing_id，后面要写入文件
    listing_id = df_test['listing_id'].values

    #数据归一化
    en= StandardScaler()
    en.fit(df_train.values)
    df_train = en.transform(df_train.values)
    df_test = en.transform(df_test.values)

    print("第四步。。。。。。。。。。。。。。")
    #开始训练，选择模型KNN
    #预测金额
    predict_amt = Model_cross_validation(df_train,label_repay_amt,df_test)

    #预测日期，把预测金额也作为属性特征
    df_train = pd.DataFrame(df_train,index = train_index,columns=train_columns)
    df_train['repay_amt'] = label_repay_amt
    df_test = pd.DataFrame(df_test,index=test_index,columns=test_columns)
    df_test['repay_amt'] = predict_amt
    #进行数据归一化
    en = StandardScaler()
    en.fit(df_train.values)
    df_train = en.transform(df_train.values)
    df_test = en.transform(df_test.values)

    #预测日期
    predict_date = Model_cross_validation(df_train, label_repay_date, df_test)

    print("第五步。。。。。。。。。。。。。。")
    #构建写入的表
    df_write = pd.DataFrame([],index = test_index,columns=['listing_id','repay_date','repay_amt'])
    df_write['listing_id'] = listing_id
    df_write['repay_date']= predict_date
    df_write['repay_amt'] = predict_amt
    df_write['auditing_date'] = df_test_auditing_date
    df_write['auditing_date'] = pd.to_datetime(df_write.auditing_date)
    #复原日期
    for i in range(len(predict_date)):
        delta = datetime.timedelta(days =predict_date[i])
        df_write.loc[i,'repay_date'] = (df_write.loc[i,'auditing_date'] + delta).date().__str__()
    df_write = df_write.drop(['auditing_date'],axis=1)
    df_write.to_csv('result.csv')
    #金额复原


def Model_cross_validation(X,y,test):
    slr= KNeighborsRegressor(weights='distance',n_neighbors=20,p=2,metric='minkowski')
    slr.fit(X,y)
    predict=slr.predict(test)
    return predict

if __name__ == "__main__":
    # df_train,df_test = train_test_data()
    # #与用户画像标签链接
    # user_taglist = user_taglist()
    # user_taglist = user_taglist.drop(['insertdate'],axis=1)
    # df_train_user_taglist = pd.merge(df_train,user_taglist,on='user_id',how='left')
    # df_test_user_taglist = pd.merge(df_test, user_taglist, on='user_id', how='left')
    # print(df_train_user_taglist.shape)
    # #与标的属性表链接
    # listing_info = listing_info()
    # listing_info = listing_info.drop(["auditing_date"],axis =1)
    # df_train_user_taglist_listing_info = pd.merge(df_train_user_taglist,listing_info,on = ['user_id','listing_id'],how = 'left')
    # df_test_user_taglist_listing_info = pd.merge(df_test_user_taglist, listing_info, on=['user_id', 'listing_id'],how='left')
    # #与借款用户基础信息表链接
    # user_info = user_info()
    # user_info = user_info.drop(['reg_mon','cell_province',"insertdate",'id_province','id_city'],axis=1)
    # df_train_user_taglist_listing_info_user_info = pd.merge(df_train_user_taglist_listing_info,user_info,on ='user_id',how = 'left')
    # df_test_user_taglist_listing_info_user_info = pd.merge(df_test_user_taglist_listing_info, user_info, on='user_id', how='left')
    # print(df_train_user_taglist_listing_info_user_info.shape)
    # #属性日期转换
    # print(df_train_user_taglist_listing_info_user_info.columns)
    # print('这是主线程：', threading.current_thread().name)
    # #创建子线程
    # thread_test = threading.Thread(target= run,kwargs={'df_test_user_taglist_listing_info_user_info':df_test_user_taglist_listing_info_user_info})
    # thread_test.start()
    # #x训练集
    # df_train_user_taglist_listing_info_user_info['auditing_date'] = pd.to_datetime(df_train_user_taglist_listing_info_user_info.auditing_date)
    # df_train_user_taglist_listing_info_user_info['repay_date'] = pd.to_datetime(df_train_user_taglist_listing_info_user_info.repay_date)
    # df_train_user_taglist_listing_info_user_info['due_date'] = pd.to_datetime(df_train_user_taglist_listing_info_user_info.due_date)
    # df_train_user_taglist_listing_info_user_info['repay_date'] = df_train_user_taglist_listing_info_user_info['repay_date'] - df_train_user_taglist_listing_info_user_info['auditing_date']
    # df_train_user_taglist_listing_info_user_info['due_date'] = df_train_user_taglist_listing_info_user_info['due_date'] - df_train_user_taglist_listing_info_user_info['auditing_date']
    # f = lambda x:str(x)[0:2]
    # df_train_user_taglist_listing_info_user_info['repay_date'] = df_train_user_taglist_listing_info_user_info['repay_date'].map(f)
    # df_train_user_taglist_listing_info_user_info['due_date'] = df_train_user_taglist_listing_info_user_info['due_date'].map(f)
    # #保存下训练和测试数据。下次运行就可以不执行前面的步骤
    # df_train_user_taglist_listing_info_user_info.to_csv("train_data.csv")
    # #主线程等待子线程结束
    # thread_test.join()
    train()