import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

#导入减少内存函数
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
#读入数据
user_info=pd.read_csv('data_format1/user_info_format1.csv')
user_info=reduce_mem_usage(user_info)
user_log=pd.read_csv('data_format1/user_log_format1.csv')
user_log=reduce_mem_usage(user_log)
train= pd.read_csv('data_format1/train_format1.csv')
train=reduce_mem_usage(train)
train.rename(columns={'merchant_id': 'seller_id'}, inplace=True)  #统一列名
test= pd.read_csv('data_format1/test_format1.csv')
test=reduce_mem_usage(test)
test.rename(columns={'merchant_id': 'seller_id'}, inplace=True)

#用每个商家的品牌众数填补nan
missingIndex = user_log[user_log.brand_id.isnull()].index
sellerMode = user_log.groupby(['seller_id']).apply(lambda x: x.brand_id.mode()[0]).reset_index()
pickUP = user_log.loc[missingIndex]
pickUP = pd.merge(pickUP, sellerMode, how='left', on=['seller_id'])[0].astype('float32')
pickUP.index = missingIndex
user_log.loc[missingIndex, 'brand_id'] = pickUP
del pickUP, sellerMode, missingIndex
gc.collect()

#中位数补年龄  众数补性别
user_info.age_range.fillna(user_info.age_range.median(),inplace=True)#年龄用中位数填充
user_info.gender.fillna(user_info.gender.mode()[0],inplace=True)# 性别用众数填充
df_age = pd.get_dummies(user_info.age_range,prefix='age')# 对age进行哑编码
df_sex = pd.get_dummies(user_info.gender)# 对gender进行哑编码并改变列名
df_sex.rename(columns={0:'female',1:'male',2:'unknown'},inplace=True)
user_info = pd.concat([user_info.user_id, df_age, df_sex], axis=1)# 整合user信息
del df_age,df_sex
gc.collect()

#以user_id为基数讨论特征
def get_userInfo_feat(user_log, user_info):
    logs = user_log
    userInfo = user_info
    # 提取全部的原始行为数据...
    print('-->Getting user logs<-- is finished...')
    totalActions = logs[["user_id", "action_type"]] #提取action
    df = pd.get_dummies(totalActions['action_type'], prefix='userTotalAction')  # 对action进行one hot
    totalActions = pd.concat([totalActions, df], axis=1).groupby(['user_id'], as_index=False).sum()  # 获得每个用户action的情况
    del df
    gc.collect()
    totalActions.drop("action_type", axis=1, inplace=True)

    # 转化比率  点击、购物车、收藏
    totalActions['userTotalAction_0_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(
        totalActions['userTotalAction_0'])
    totalActions['userTotalAction_1_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(
        totalActions['userTotalAction_1'])
    totalActions['userTotalAction_3_ratio'] = np.log1p(totalActions['userTotalAction_2']) - np.log1p(
        totalActions['userTotalAction_3'])
    # 归一化以上3个比率
    cols = [i for i in totalActions.columns.tolist() if i not in ['user_id']]
    for col in cols:
        s = (totalActions[col] - totalActions[col].min()) / (totalActions[col].max() - totalActions[col].min())
        totalActions = totalActions.drop([col], axis=1)
        totalActions[col] = s
    print('-->Counting total numbers of clicking,addCart,buying,favor in various Users<-- are finished...')
    # 拼接
    userInfo = pd.merge(userInfo, totalActions, how='left', on=['user_id'])
    del totalActions
    gc.collect()

    typeCounts = logs[["user_id", "seller_id", "cat_id", "item_id", "brand_id", "time_stamp"]]
    # 每个用户买了多少种商品
    typeCount_result = typeCounts.groupby(['user_id'])['cat_id'].nunique()
    # 每个用户买了多少类品牌
    typeCount_result = pd.concat([typeCount_result, typeCounts.groupby(['user_id'])['brand_id'].nunique()], axis=1)
    # 每个用户活跃的天数
    typeCount_result = pd.concat([typeCount_result, typeCounts.groupby(['user_id'])['time_stamp'].nunique()],axis=1)
    typeCount_result.rename(columns={'cat_id': 'cat_counts', 'brand_id': 'brand_counts', 'time_stamp': 'active_days'}, inplace=True)
    typeCount_result.reset_index(inplace=True)
    # 对以上三个数据归一化
    typeCount_result['cat_counts'] = mms.fit_transform(typeCount_result[['cat_counts']])
    typeCount_result['brand_counts'] = mms.fit_transform(typeCount_result[['brand_counts']])
    typeCount_result['active_days'] = mms.fit_transform(typeCount_result[['active_days']])
    print('-->Counting total numbers of items, cat, brand and action_days in various Users<-- are finished...')
    # 拼接
    userInfo = pd.merge(userInfo, typeCount_result, how='left', on=['user_id'])
    del typeCount_result, typeCounts
    gc.collect()

    ## 统计双十一之前，购买过的商家数量
    repeatSellerCount = logs[["user_id", "seller_id", "time_stamp", "action_type"]]
    # 筛选出双11前购买的数据
    repeatSellerCount = repeatSellerCount[(repeatSellerCount.action_type == 2) & (repeatSellerCount.time_stamp < 1111)]
    # 删除重复项
    repeatSellerCount.drop_duplicates(inplace=True)
    repeatSellerCount = repeatSellerCount.groupby(['user_id', 'seller_id'])['time_stamp'].count().reset_index()
    repeatSellerCount = repeatSellerCount[repeatSellerCount.time_stamp > 1]
    # 统计每个用户购买每个商家的次数
    repeatSellerCount = repeatSellerCount.groupby(['user_id'])['seller_id'].count().reset_index()
    repeatSellerCount.rename(columns={'seller_id': 'repeat_seller_count'}, inplace=True)
    cols = ['repeat_seller_count']
    # 对这条数据归一化
    for col in cols:
        s = (repeatSellerCount[col] - repeatSellerCount[col].min()) / (
                    repeatSellerCount[col].max() - repeatSellerCount[col].min())
        repeatSellerCount = repeatSellerCount.drop([col], axis=1)
        repeatSellerCount[col] = s
    # repeatSellerCount['repeat_seller_count'] = mms.fit_transform(repeatSellerCount[['repeat_seller_count']])
    print('-->Counting seller number of repeat buying in various Users<-- are finished...')
    userInfo = pd.merge(userInfo, repeatSellerCount, how='left', on=['user_id'])
    userInfo.repeat_seller_count.fillna(0, inplace=True)
    del repeatSellerCount
    gc.collect()

    ## 用户除了目标商家外，对其他重复回购商店的点击，加入购物车，购买，收藏的平均值
    logs = user_log
    trainData = pd.read_csv('data_format1/train_format1.csv')
    trainData.rename(columns={'merchant_id': 'seller_id'}, inplace=True)
    testData = pd.read_csv('data_format1/test_format1.csv')
    testData.rename(columns={'merchant_id': 'seller_id'}, inplace=True)
    targetIndex = pd.concat([trainData[['user_id', 'seller_id']], testData[['user_id', 'seller_id']]],ignore_index=True)
    removeLogs = pd.merge(targetIndex, logs, on=['user_id', 'seller_id'])
    del trainData, testData, targetIndex
    gc.collect()
    # 选出除了目标商家的数据
    logs.drop(removeLogs.index, inplace=True)

    ## 每个用户（除去目标用户对目标商家的记录）对不同商家的行为总数
    actionsRate = logs[["user_id", "seller_id", "action_type"]]
    df = pd.get_dummies(actionsRate['action_type'], prefix='repeat_actionRate') #对行为one hot
    actionsRate = pd.concat([actionsRate, df], axis=1).groupby(['user_id', 'seller_id'], as_index=False).sum()
    del df
    gc.collect()
    actionsRate.drop("action_type", axis=1, inplace=True)

    ## 找出每个用户对回购商家的行为总数
    repeat = logs[["user_id", "seller_id", "time_stamp", "action_type"]]
    repeat = repeat[(repeat.action_type == 2) & (repeat.time_stamp < 1111)]
    repeat.drop_duplicates(inplace=True)
    repeat = repeat.groupby(['user_id', 'seller_id'])['time_stamp'].count().reset_index()
    repeat = repeat[repeat.time_stamp > 1]
    actionsRate = pd.merge(actionsRate, repeat, on=['user_id', 'seller_id'])
    actionsRate.drop("time_stamp", axis=1, inplace=True)
    del repeat
    gc.collect()
    actionsRate = actionsRate.groupby(['user_id'])[['repeat_actionRate_0', 'repeat_actionRate_1', 'repeat_actionRate_2','repeat_actionRate_3']].mean().reset_index()
    # 归一化
    actionsRate['repeat_actionRate_0'] = mms.fit_transform(actionsRate[['repeat_actionRate_0']])
    actionsRate['repeat_actionRate_1'] = mms.fit_transform(actionsRate[['repeat_actionRate_1']])
    actionsRate['repeat_actionRate_2'] = mms.fit_transform(actionsRate[['repeat_actionRate_2']])
    actionsRate['repeat_actionRate_3'] = mms.fit_transform(actionsRate[['repeat_actionRate_3']])
    userInfo = pd.merge(userInfo, actionsRate, how='left', on=['user_id'])
    userInfo.repeat_actionRate_0.fillna(0, inplace=True)
    userInfo.repeat_actionRate_1.fillna(0, inplace=True)
    userInfo.repeat_actionRate_2.fillna(0, inplace=True)
    userInfo.repeat_actionRate_3.fillna(0, inplace=True)
    del actionsRate
    gc.collect()

    ## 统计每月的点击次数，每月的加入购物次数，每月的购买次数，每月的收藏次数
    monthActionsCount = logs[["user_id", "time_stamp", "action_type"]]
    result = list()
    for i in range(5, 12):
        start = int(str(i) + '00')
        end = int(str(i) + '30')
        example = monthActionsCount[(monthActionsCount.time_stamp >= start) & (monthActionsCount.time_stamp < end)]
        df = pd.get_dummies(example['action_type'], prefix='%d_Action' % i)
        example.loc[:, 'time_stamp'] = example.time_stamp.apply(lambda x: int(str(x)[0]) if len(str(x)) == 3 else int(str(x)[:2]))
        result.append(pd.concat([example, df], axis=1).groupby(['user_id', 'time_stamp'], as_index=False).sum())
    for i in range(0, 7):
        userInfo = pd.merge(userInfo, result[i], how='left', on=['user_id'])
        userInfo.fillna(0, inplace=True)

    # pickle.dump(userInfo, open(filePath, 'wb'))
    return userInfo
userInfo = get_userInfo_feat(user_log, user_info)

#以商户为基数讨论特征
def get_sellerInfo_feat(user_log):
    logs = user_log
    ## 统计每个商户的商品，种类，品牌总数
    itemNumber = logs.groupby(['seller_id'])['item_id'].nunique().reset_index()
    catNumber = logs.groupby(['seller_id'])['cat_id'].nunique().reset_index()
    brandNumber = logs.groupby(['seller_id'])['brand_id'].nunique().reset_index()
    itemNumber.rename(columns={'item_id': 'item_number'}, inplace=True)
    catNumber.rename(columns={'cat_id': 'cat_number'}, inplace=True)
    brandNumber.rename(columns={'brand_id': 'brand_number'}, inplace=True)
    print('-->Counting numbers of product, category, brand in various sellers<-- are finished...')

    ## 统计商户重复购买买家总数量
    repeatPeoCount = logs[(logs.time_stamp < 1111) & (logs.action_type == 2)]
    repeatPeoCount = repeatPeoCount.groupby(['seller_id'])['user_id'].value_counts().to_frame()
    repeatPeoCount.rename(columns={'user_id': 'Buy_Number'}, inplace=True)
    repeatPeoCount.reset_index(inplace=True)
    repeatPeoCount = repeatPeoCount[repeatPeoCount.Buy_Number > 1]
    repeatPeoCount = repeatPeoCount.groupby(['seller_id']).apply(lambda x: len(x.user_id)).reset_index()
    repeatPeoCount = pd.merge(pd.DataFrame({'seller_id': range(1, 4996, 1)}), repeatPeoCount, how='left',on=['seller_id']).fillna(0)
    repeatPeoCount.rename(columns={0: 'repeatBuy_peopleNumber'}, inplace=True)
    print('-->Counting numbers of repeat buying buyers in various sellers<-- are finished...')

    # 统计商户重复点击买家总数量
    repeatPeoCount1 = logs[(logs.time_stamp < 1111) & (logs.action_type == 0)]
    repeatPeoCount1 = repeatPeoCount1.groupby(['seller_id'])['user_id'].value_counts().to_frame()
    repeatPeoCount1.rename(columns={'user_id': 'click_Number'}, inplace=True)
    repeatPeoCount1.reset_index(inplace=True)
    repeatPeoCount1 = repeatPeoCount1[repeatPeoCount1.click_Number > 1]
    repeatPeoCount1 = repeatPeoCount1.groupby(['seller_id']).apply(lambda x: len(x.user_id)).reset_index()
    repeatPeoCount1 = pd.merge(pd.DataFrame({'seller_id': range(1, 4996, 1)}), repeatPeoCount1, how='left',on=['seller_id']).fillna(0)
    repeatPeoCount1.rename(columns={0: 'repeatclick_peopleNumber'}, inplace=True)
    print('-->Counting numbers of click buying buyers in various sellers<-- are finished...')

    # 统计商户重复加入购物车买家总数量
    repeatPeoCount3 = logs[(logs.time_stamp < 1111) & (logs.action_type == 1)]
    repeatPeoCount3 = repeatPeoCount3.groupby(['seller_id'])['user_id'].value_counts().to_frame()
    repeatPeoCount3.rename(columns={'user_id': 'cart_Number'}, inplace=True)
    repeatPeoCount3.reset_index(inplace=True)
    repeatPeoCount3 = repeatPeoCount3[repeatPeoCount3.cart_Number > 1]
    repeatPeoCount3 = repeatPeoCount3.groupby(['seller_id']).apply(lambda x: len(x.user_id)).reset_index()
    repeatPeoCount3 = pd.merge(pd.DataFrame({'seller_id': range(1, 4996, 1)}), repeatPeoCount3, how='left',on=['seller_id']).fillna(0)
    repeatPeoCount3.rename(columns={0: 'repeatcart_peopleNumber'}, inplace=True)
    print('-->Counting numbers of cart buying buyers in various sellers<-- are finished...')

    #统计商户重复点赞买家总数量
    repeatPeoCount2 = logs[(logs.time_stamp < 1111) & (logs.action_type == 3)]
    repeatPeoCount2 = repeatPeoCount2.groupby(['seller_id'])['user_id'].value_counts().to_frame()
    repeatPeoCount2.rename(columns={'user_id': 'fav_Number'}, inplace=True)
    repeatPeoCount2.reset_index(inplace=True)
    repeatPeoCount2 = repeatPeoCount2[repeatPeoCount2.fav_Number > 1]
    repeatPeoCount2 = repeatPeoCount2.groupby(['seller_id']).apply(lambda x: len(x.user_id)).reset_index()
    repeatPeoCount2 = pd.merge(pd.DataFrame({'seller_id': range(1, 4996, 1)}), repeatPeoCount2, how='left',on=['seller_id']).fillna(0)
    repeatPeoCount2.rename(columns={0: 'repeatfav_peopleNumber'}, inplace=True)
    print('-->Counting numbers of fav buying buyers in various sellers<-- are finished...')

    ##统计被点击，被加入购物车，被购买，被收藏次数
    ###统计被点击购买转化率，被加入购物车购买转化率，被收藏次数购买转化率
    sellers = logs[["seller_id", "action_type"]]
    df = pd.get_dummies(sellers['action_type'], prefix='seller')
    sellers = pd.concat([sellers, df], axis=1).groupby(['seller_id'], as_index=False).sum()
    sellers.drop("action_type", axis=1, inplace=True)
    #转化率
    sellers['seller_0_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_0'])
    sellers['seller_1_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_1'])
    sellers['seller_3_ratio'] = np.log1p(sellers['seller_2']) - np.log1p(sellers['seller_3'])
    print('-->Counting numbers of clicking, addCart, buying,subcribes in various sellers<-- are finished...')

    #统计每个商户被点击的人数，被加入购物车的人数，被购买的人数，被收藏的人数
    peoCount = logs[["user_id", "seller_id", "action_type"]]
    df = pd.get_dummies(peoCount['action_type'], prefix='seller_peopleNumber')
    peoCount = pd.concat([peoCount, df], axis=1)
    peoCount.drop("action_type", axis=1, inplace=True)
    peoCount.drop_duplicates(inplace=True)
    df1 = peoCount.groupby(['seller_id']).apply(lambda x: x.seller_peopleNumber_0.sum())
    df2 = peoCount.groupby(['seller_id']).apply(lambda x: x.seller_peopleNumber_1.sum())
    df3 = peoCount.groupby(['seller_id']).apply(lambda x: x.seller_peopleNumber_2.sum())
    df4 = peoCount.groupby(['seller_id']).apply(lambda x: x.seller_peopleNumber_3.sum())
    peoCount = pd.concat([df1, df2, df3, df4], axis=1).reset_index()
    peoCount.rename(columns={0: 'seller_peopleNum_0', 1: 'seller_peopleNum_1', 2: 'seller_peopleNum_2',3: 'seller_peopleNum_3'}, inplace=True)
    print('-->Counting numbers of product, category, brand in various sellers<-- are finished...')

    #拼接
    sellers = pd.merge(sellers, peoCount, on=['seller_id'])
    sellers = pd.merge(sellers, itemNumber, on=['seller_id'])
    sellers = pd.merge(sellers, catNumber, on=['seller_id'])
    sellers = pd.merge(sellers, brandNumber, on=['seller_id'])
    sellers = pd.merge(sellers, repeatPeoCount, on=['seller_id'])
    sellers = pd.merge(sellers, repeatPeoCount1, on=['seller_id'])
    sellers = pd.merge(sellers, repeatPeoCount2, on=['seller_id'])
    sellers = pd.merge(sellers, repeatPeoCount3, on=['seller_id'])

    #归一化
    sellers['seller_0'] = mms.fit_transform(sellers[['seller_0']])
    sellers['seller_1'] = mms.fit_transform(sellers[['seller_1']])
    sellers['seller_2'] = mms.fit_transform(sellers[['seller_2']])
    sellers['seller_3'] = mms.fit_transform(sellers[['seller_3']])
    sellers['seller_peopleNum_0'] = mms.fit_transform(sellers[['seller_peopleNum_0']])
    sellers['seller_peopleNum_1'] = mms.fit_transform(sellers[['seller_peopleNum_1']])
    sellers['seller_peopleNum_2'] = mms.fit_transform(sellers[['seller_peopleNum_2']])
    sellers['seller_peopleNum_3'] = mms.fit_transform(sellers[['seller_peopleNum_3']])
    sellers['item_number'] = mms.fit_transform(sellers[['item_number']])
    sellers['cat_number'] = mms.fit_transform(sellers[['cat_number']])
    sellers['brand_number'] = mms.fit_transform(sellers[['brand_number']])
    sellers['repeatBuy_peopleNumber'] = mms.fit_transform(sellers[['repeatBuy_peopleNumber']])
    sellers['repeatclick_peopleNumber'] = mms.fit_transform(sellers[['repeatclick_peopleNumber']])
    sellers['repeatcart_peopleNumber'] = mms.fit_transform(sellers[['repeatcart_peopleNumber']])
    sellers['repeatfav_peopleNumber'] = mms.fit_transform(sellers[['repeatfav_peopleNumber']])

    del df1, df2, df3, df4, itemNumber, catNumber, brandNumber, df, logs, mms, peoCount, repeatPeoCount,repeatPeoCount1,repeatPeoCount2,repeatPeoCount3
    gc.collect()
    # pickle.dump(sellers, open(filePath, 'wb'))
    return sellers
sellers = get_sellerInfo_feat(user_log)

#以user和sell共同为基数讨论特征
def get_userSellerActions_feat(user_log):
        logs = user_log
        print('-->Getting user logs<-- is finished...')
        trainData = pd.read_csv('data_format1/train_format1.csv')
        trainData.rename(columns={'merchant_id': 'seller_id'}, inplace=True)
        testData = pd.read_csv('data_format1/test_format1.csv')
        testData.rename(columns={'merchant_id': 'seller_id'}, inplace=True)
        targetIndex = pd.concat([trainData[['user_id', 'seller_id']], testData[['user_id', 'seller_id']]],ignore_index=True)
        logs = pd.merge(targetIndex, logs, on=['user_id', 'seller_id'])  #合并train和test
        del trainData, testData, targetIndex
        gc.collect()

        #统计用户对预测的商店的行为特征，例如点击，加入购物车，购买，收藏的总次数,以及各种转化率
        df_result = logs[["user_id", "seller_id", "action_type"]]
        df = pd.get_dummies(df_result['action_type'], prefix='userSellerAction')
        df_result = pd.concat([df_result, df], axis=1).groupby(['user_id', 'seller_id'], as_index=False).sum()
        del df
        gc.collect()
        df_result.drop("action_type", axis=1, inplace=True)
        df_result['userSellerAction_0_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_0'])
        df_result['userSellerAction_1_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_1'])
        df_result['userSellerAction_3_ratio'] = np.log1p(df_result['userSellerAction_2']) - np.log1p(df_result['userSellerAction_3'])

        #统计用户对预测商店，所点击的总天数
        clickDays = logs[logs.action_type == 0]
        clickDays = clickDays[["user_id", "seller_id", "time_stamp", "action_type"]]
        clickDays = clickDays.groupby(['user_id', 'seller_id']).apply(lambda x: x.time_stamp.nunique()).reset_index()
        clickDays.rename(columns={0: 'click_days'}, inplace=True)
        df_result = pd.merge(df_result, clickDays, how='left', on=['user_id', 'seller_id'])
        df_result.click_days.fillna(0, inplace=True)
        del clickDays
        gc.collect()

        #购买商品种类数量，点击商品种类数量
        catTypeCount = logs[["user_id", "seller_id", "cat_id", "action_type"]]
        catTypeCount = catTypeCount[(catTypeCount.action_type == 0) | (catTypeCount.action_type == 2)]
        cat_result = catTypeCount[catTypeCount.action_type == 2].groupby(['user_id', 'seller_id']).apply(lambda x: x.cat_id.nunique()).reset_index()
        cat_result.rename(columns={0: 'buy_catType_count'}, inplace=True)
        cat_result = pd.merge(cat_result,catTypeCount[catTypeCount.action_type == 0].groupby(['user_id', 'seller_id']).apply(lambda x: x.cat_id.nunique()).reset_index(), how='left', on=['user_id', 'seller_id'])
        cat_result.rename(columns={0: 'click_catType_count'}, inplace=True)
        cat_result.click_catType_count.fillna(0, inplace=True)
        df_result = pd.merge(df_result, cat_result, how='left', on=['user_id', 'seller_id'])
        del catTypeCount, cat_result
        gc.collect()
        # pickle.dump(df_result, open(filePath, 'wb'))
        return df_result
usersell= get_userSellerActions_feat(user_log)

#拼接
train= pd.merge(train, userInfo, how='left', on=['user_id'])
test= pd.merge(test, userInfo, how='left', on=['user_id'])
train= pd.merge(train, sellers, how='left', on=['seller_id'])
test= pd.merge(test, sellers, how='left', on=['seller_id'])
train= pd.merge(train, usersell, how='left', on=['user_id', 'seller_id'])
test= pd.merge(test, usersell, how='left', on=['user_id', 'seller_id'])
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
#保存
# train.to_csv('trainset.csv',index=False)
# test.to_csv('testset.csv',index=False)
# train=pd.read_csv('trainset.csv')
# test=pd.read_csv('testset.csv')

#提取特征
features= [c for c in train.columns.tolist() if c not in ['user_id', 'seller_id','label','prob']]

#构建单模型 分数在0.685左右
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
train_x = train[features]
train_y = train['label']
test_x=test[features]
model=lgb.LGBMClassifier(learning_rate=0.01,n_estimators=1500,random_state=2019)
for train_idx, val_idx in kfold.split(train_x):
    train_x1 = train_x.loc[train_idx]
    train_y1 = train_y.loc[train_idx]
    test_x1 = train_x.loc[val_idx]
    test_y1 = train_y.loc[val_idx]
    #,(vali_x,vali_y)
    model.fit(train_x1, train_y1,eval_set=[(train_x1, train_y1),(test_x1, test_y1)],eval_metric='auc')
    test['prob'] = test['prob'] + model.predict_proba(test_x)[:,1]
test['prob'] = test['prob'] / 10
test.rename(columns={'seller_id':'merchant_id'}, inplace=True)
test[['user_id','merchant_id','prob']].to_csv('result1.csv',index=False)

#构建stacking模型 分数上升了0.001
train_x = train[features]
train_y = train['label']
test_x=test[features]
clf1= xgb.XGBRFClassifier(learning_rate=0.01,n_estimators=1500,random_state=2019)
clf2= lgb.LGBMClassifier(learning_rate=0.01,n_estimators=1500,random_state=2019)
dtc=DecisionTreeClassifier()
sclf = StackingClassifier(classifiers=[clf1,clf2], meta_classifier=dtc)
for clf, label in zip([clf1,clf2,sclf], ['xgb','lgb','StackingClassifier']):
    scores = model_selection.cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))




