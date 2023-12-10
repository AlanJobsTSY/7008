import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
folder_path = r'.'

anames = ['id', 'name', 'url', 'pictureURL']
artists = pd.read_table(os.path.join(folder_path, 'artists.dat'), sep='\\t', header=None, names=anames, engine='python', skiprows=1, encoding='ISO-8859-1')

tnames = ['tagID', 'tagValue']
tags = pd.read_table(os.path.join(folder_path, 'tags.dat'), sep='\\t', header=None, names=tnames, engine='python', skiprows=1, encoding='ISO-8859-1')

uanames = ['userID', 'artistID', 'weight']
user_artists = pd.read_table(os.path.join(folder_path, 'user_artists.dat'), sep='\\t', header=None, names=uanames, engine='python', skiprows=1, encoding='ISO-8859-1')

utnames = ['userID', 'artistID', 'tagID', 'day', 'month', 'year']
user_taggedartists = pd.read_table(os.path.join(folder_path, 'user_taggedartists.dat'), sep='\\t', header=None, names=utnames, engine='python', skiprows=1, encoding='ISO-8859-1')

uttnames = ['userID', 'artistID', 'tagID', 'timestamp']
user_taggedartists_timestamps = pd.read_table(os.path.join(folder_path, 'user_taggedartists-timestamps.dat'), sep='\\t', header=None, names=uttnames, engine='python', skiprows=1, encoding='ISO-8859-1')

ufnames = ['userID', 'friendID']
user_friends = pd.read_table(os.path.join(folder_path, 'user_friends.dat'), sep='\\t', header=None, names=ufnames, engine='python', skiprows=1, encoding='ISO-8859-1')

print(user_artists[:10])

from sklearn.model_selection import train_test_split

# 获取所有唯一的用户ID
unique_users = user_artists['userID'].unique()

# 初始化训练集和测试集的空列表
train_data_list = []
test_data_list = []

# 分割每个用户的数据
for user_id in unique_users:
    # 获取当前用户的数据
    user_data = user_artists[user_artists['userID'] == user_id]

    # 检查用户数据是否足够多
    if len(user_data) > 1:
        # 分割当前用户的数据为训练集和测试集
        train_data, test_data = train_test_split(user_data, test_size=0.2)  # 80% 训练集，20% 测试集

        # 将分割后的数据加入到对应的列表中
        train_data_list.append(train_data)
        test_data_list.append(test_data)
# 使用 concat 方法将列表中的数据拼接为DataFrame
train_set = pd.concat(train_data_list)
test_set = pd.concat(test_data_list)

# 确认分割后的数据集大小
print("训练集大小:", len(train_set))
print("测试集大小:", len(test_set))
# 导出训练集为文本文件
train_set.to_csv('train.txt', sep='\t', index=False,header=False)
# 导出测试集为文本文件
test_set.to_csv('test.txt', sep='\t', index=False,header=False)

user_friends.to_csv('trustnetwork.txt', sep='\t', index=False,header=False)