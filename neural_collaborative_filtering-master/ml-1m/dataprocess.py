import pandas as pd

# Read the rating.dat file
ratings_data = pd.read_csv('ratings.dat', sep='::', engine='python', header=None)
ratings_data.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Create a new file "ml-1m.train.rating" and write the data in the desired format
with open('ml-1m.train.rating', 'w') as file:
    for index, row in ratings_data.iterrows():
        user_id = row['UserID']
        movie_id = row['MovieID']
        rating = row['Rating']
        timestamp = row['Timestamp']
        # Write the data in the desired format: UserID::MovieID::Rating::Timestamp
        file.write(f"{user_id}\t{movie_id}\t{rating}\t{timestamp}\n")