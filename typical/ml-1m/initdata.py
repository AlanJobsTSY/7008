import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
folder_path = r'' 
unames = ['user_id', 'gender', 'age', 'occupation', 'zip'] 
users = pd.read_table(os.path.join(folder_path,'users.dat'), sep='::', header=None, names=unames)
rnames = ['user_id', 'movie_id', 'rating',	'timestamp'] 
ratings = pd.read_table(os.path.join(folder_path,'ratings.dat'), sep = '::', header=None, names=rnames) 
mnames = ['movie_id', 'title', 'genres'] 
movies = pd.read_table(os.path.join(folder_path,'movies.dat'),sep='::',header=None,	names=mnames,engine='python', encoding='ISO-8859-1')
print(ratings[:5])
# 按照 user_id 分组并过滤 rating 大于等于 3 的数据
filtered_ratings = ratings[ratings['rating'] >= 3].groupby('user_id')['movie_id'].apply(list).reset_index()
print(filtered_ratings.head())

with open('train.txt', 'w') as f_train, open('test.txt', 'w') as f_test, open('user_map','w')as f_map:
    for index, row in filtered_ratings.iterrows():
        user_id = row['user_id']
        movie_list = row['movie_id']
        train_size = int(len(movie_list) * 0.8)  # 80% for training
        if train_size>=1:
            train_data = ' '.join(str(movie_id) for movie_id in movie_list[:train_size])
            test_data = ' '.join(str(movie_id) for movie_id in movie_list[train_size:])
        else:
            train_data = ' '.join(str(movie_id) for movie_id in movie_list[:])
            test_data = ' '.join(str(movie_id) for movie_id in movie_list[:])
        f_train.write(f"{index} {train_data}\n")
        f_test.write(f"{index} {test_data}\n")
        f_map.write(f"{index} {user_id}\n")



