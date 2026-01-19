import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic structure
print("Table Structure:")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())

# Summary statistics for numerical columns
print("\nSummary of numerical columns (first vote, place):")
print(df[['first vote', 'place']].describe())

# Correlation between first vote and place (to identify trends)
correlation = df['first vote'].corr(df['place'])
print(f"\nCorrelation between first vote and place: {correlation:.2f}")

# Highlight the song with highest first vote and its place
top_vote_song = df.loc[df['first vote'].idxmax()]
print(f"\nNotable anomaly: Song with highest first vote - '{top_vote_song['song']}' by {top_vote_song['artist']} with {top_vote_song['first vote']} votes, placed {top_vote_song['place']}.")