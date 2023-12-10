import os
import pandas as pd
from sklearn.model_selection import train_test_split 
import shutil
from pathlib import Path

# Changing the current working directory to where the raw data is located
os.chdir('/content/drive/My Drive/TransR/Data/Raw/')

# these two file is just for pairing id with name
artists = pd.read_csv('./lastfm_raw/artists.dat', sep='\t')
tags = pd.read_csv('./lastfm_raw/tags.dat', sep='\t', encoding='ISO-8859-1')

# these 3 files describe 3 different type of relationships
user_artists = pd.read_csv('./lastfm_raw/user_artists.dat', sep='\t')
user_taggedartists = pd.read_csv('./lastfm_raw/user_taggedartists.dat', sep='\t')
user_friends = pd.read_csv('./lastfm_raw/user_friends.dat', sep='\t')

"""### Remove unnecessary features"""

artists.drop(['url', 'pictureURL'], axis=1, inplace=True)
user_artists.drop(['weight'], axis=1, inplace=True)
user_taggedartists.drop(['day', 'month', 'year'], axis=1, inplace=True)

"""### Data Cleaning"""

# Clean 'user_artists' to ensure that it only contains valid 'artistID's
user_artists = user_artists[user_artists['artistID'].isin(artists['id'])]

# Clean 'user_taggedartists' to ensure it only contains valid 'artistID's and 'tagID's
user_taggedartists = user_taggedartists[
    user_taggedartists['artistID'].isin(artists['id']) &
    user_taggedartists['tagID'].isin(tags['tagID'])
]
# Now user_artists and user_taggedartists should only contain IDs that are present in artists and tags respectively

# Count unique users
unique_users = pd.unique(pd.concat([user_artists['userID'], user_friends['userID']]))
num_unique_users = len(unique_users)
print(f"Number of unique users: {num_unique_users}")

# Count friend relation pairs
num_friend_pairs = len(user_friends)
print(f"Number of 'friend' relation pairs: {num_friend_pairs}")

# Count unique artists
unique_artists = pd.unique(pd.concat([user_artists['artistID'], user_taggedartists['artistID']]))
num_unique_artists = len(unique_artists)
print(f"Number of unique artists: {num_unique_artists}")

# Count unique tags
unique_tags = pd.unique(user_taggedartists['tagID'])
num_unique_tags = len(unique_tags)
print(f"Number of unique tags: {num_unique_tags}")

# Count listens_to relations
num_listens_to_relations = len(user_artists)
print(f"Number of 'listens_to' relations: {num_listens_to_relations}")

# Count assigns_tag relations (unique user-tag pairs)
num_assigns_tag_relations = len(user_taggedartists[['userID', 'tagID']].drop_duplicates())
print(f"Number of 'assigns_tag' relations: {num_assigns_tag_relations}")

# Count has_tag relations (unique artist-tag pairs)
num_has_tag_relations = len(user_taggedartists[['artistID', 'tagID']].drop_duplicates())
print(f"Number of 'has_tag' relations: {num_has_tag_relations}")

# Filter out tags that are not used
tags = tags[tags['tagID'].isin(unique_tags)]

# Now 'tags' only contains tags that are actually used

"""### Convert into required data format

**Step 1:** we need to re-index all entites and relations in the required format.

**note:**  we do not have name for "User" Entities, so we just use the ID as its name.

**Important:** index from 0!! index from 1 will cause error passing through embedding layer.
"""

# Create a DataFrame for users with the old 'userID' as the index
users = pd.DataFrame({'old_id': pd.unique(user_artists['userID']), 'type': 'user'})
users = users.reset_index()
users['new_id'] = users.index   # New unique ID
users['name'] = users['new_id'].astype(str)  # Use the new unique ID as the 'name'

# Prepare the artists and tags DataFrames
artists = artists.reset_index()
artists['new_id'] = artists.index + len(users)
artists['name'] = artists['name'].astype(str)

tags = tags.reset_index()
tags['new_id'] = tags.index + len(users) + len(artists)
tags['name'] = tags['tagValue']

# Concatenate all entities into a single DataFrame with their new unique ID and names
all_entities = pd.concat([
    users[['name', 'new_id']],
    artists[['name', 'new_id']],
    tags[['name', 'new_id']]
])
# Now all_entities DataFrame will have unique IDs for all entities starting from 1

# Given relation_ids dictionary
relation_ids = {
    "listens_to": 0,
    "friends_with": 1,
    "assigns_tag": 2,
    "has_tag": 3
}

# Convert the dictionary to a pandas DataFrame
all_relations = pd.DataFrame(list(relation_ids.items()), columns=['relation', 'id'])
# Now we have a DataFrame with the relations and their corresponding IDs

"""**Step2:** We need to create entity-relation-entity triples in the required format."""

# Convert User-Artist relationships
user_artist_triples = user_artists[['userID', 'artistID']].copy()
user_artist_triples['relationID'] = relation_ids['listens_to']

# We'll use the unique tags directly associated with artists for the "has_tag" relation
artist_tag_triples = user_taggedartists[['artistID', 'tagID']].drop_duplicates().copy()
artist_tag_triples['relationID'] = relation_ids['has_tag']

# User-Tag assignment (ignoring the specific artistID)
user_tag_triples = user_taggedartists[['userID', 'tagID']].drop_duplicates().copy()
user_tag_triples['relationID'] = relation_ids['assigns_tag']

# Convert User-Friends relationships
user_friends_triples = user_friends[['userID', 'friendID']].copy()
user_friends_triples['relationID'] = relation_ids['friends_with']

# Create a mapping from old IDs to new IDs for each entity type
user_id_mapping = users.set_index('old_id')['new_id'].to_dict()
artist_id_mapping = artists.set_index('id')['new_id'].to_dict()
tag_id_mapping = tags.set_index('tagID')['new_id'].to_dict()

# Update the IDs in the user-artist relationship DataFrame
user_artist_triples['userID'] = user_artist_triples['userID'].map(user_id_mapping)
user_artist_triples['artistID'] = user_artist_triples['artistID'].map(artist_id_mapping)

# Update the IDs in the artist-tag relationship DataFrame
artist_tag_triples['artistID'] = artist_tag_triples['artistID'].map(artist_id_mapping)
artist_tag_triples['tagID'] = artist_tag_triples['tagID'].map(tag_id_mapping)

# Update the IDs in the user-tag relationship DataFrame
user_tag_triples['userID'] = user_tag_triples['userID'].map(user_id_mapping)
user_tag_triples['tagID'] = user_tag_triples['tagID'].map(tag_id_mapping)

# Update the IDs in the user-friends relationship DataFrame
user_friends_triples['userID'] = user_friends_triples['userID'].map(user_id_mapping)
user_friends_triples['friendID'] = user_friends_triples['friendID'].map(user_id_mapping)

# Changing the directory to where the processed data will be stored
os.chdir('/content/drive/My Drive/TransR/Data/Processed/lastfm/')

# Write to entity2id.txt
with open('./entity2id.txt', 'w+') as file:
    # First line is the number of entities
    file.write(f"{len(all_entities)}\n")
    # Then write all entities and their new IDs
    for index, row in all_entities.iterrows():
        file.write(f"{row['name']}\t{row['new_id']}\n")

# Write to relation2id.txt
with open('./relation2id.txt', 'w+') as file:
    # First line is the number of relations
    file.write(f"{len(all_relations)}\n")
    # Then write all relations and their IDs
    for index, row in all_relations.iterrows():
        file.write(f"{row['relation']}\t{row['id']}\n")
        
# First, rename the columns in each triple dataframe to have a consistent "entity1", "entity2", "relationID" format
user_artist_triples.rename(columns={'userID': 'entity1', 'artistID': 'entity2'}, inplace=True)
artist_tag_triples.rename(columns={'artistID': 'entity1', 'tagID': 'entity2'}, inplace=True)
user_tag_triples.rename(columns={'userID': 'entity1', 'tagID': 'entity2'}, inplace=True)
user_friends_triples.rename(columns={'userID': 'entity1', 'friendID': 'entity2'}, inplace=True)

# Combine all updated triples into one DataFrame
all_triples = pd.concat([
    user_artist_triples[['entity1', 'entity2', 'relationID']],
    artist_tag_triples[['entity1', 'entity2', 'relationID']],
    user_tag_triples[['entity1', 'entity2', 'relationID']],
    user_friends_triples[['entity1', 'entity2', 'relationID']]
])

# Remove any NaN values
all_triples.dropna(inplace=True)
all_triples = all_triples.astype(int)

# Shuffle the DataFrame
all_triples = all_triples.sample(frac=1, random_state=7008).reset_index(drop=True)

# Split the data into train, validation, and test sets
train, temp = train_test_split(all_triples, test_size=0.2, random_state=7008)
valid, test = train_test_split(temp, test_size=0.5, random_state=7008)

# Save the data sets to txt files
def save_to_txt(df, file_path):
    with open(file_path, 'w+') as f:
        f.write(f"{len(df)}\n")  # First line: number of triples
        df.to_csv(f, sep=' ', index=False, header=False)

save_to_txt(train, './train2id.txt')
save_to_txt(valid, './valid2id.txt')
save_to_txt(test, './/test2id.txt')

# """
# Copy the processed data into needed file
# """

# Define the source and destination paths
source_path = Path('/content/drive/My Drive/TransR/Data/Processed/lastfm')
destination_path = Path('/content/drive/My Drive/TransR/Model/OpenKE/benchmarks/lastfm')

# Ensure the destination directory does not already exist
if destination_path.exists():
    print(f"The directory {destination_path} already exists.")
else:
    # Copy the directory
    shutil.copytree(source_path, destination_path)
    print(f"Directory copied from {source_path} to {destination_path}")
