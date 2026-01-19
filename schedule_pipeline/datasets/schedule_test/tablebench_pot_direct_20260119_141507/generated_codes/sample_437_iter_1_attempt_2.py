import pandas as pd
df = pd.read_csv('table.csv')
print("Table Structure:")
print(df.info())
print("\nSummary of first vote and place:")
print(df[['first vote', 'place']].describe())
max_vote_row = df.loc[df['first vote'].idxmax()]
print(f"\nNotable anomaly: '{max_vote_row['song']}' by {max_vote_row['artist']} had the highest first vote ({max_vote_row['first vote']}) and placed {max_vote_row['place']}.")
high_vote_winners = df[(df['first vote'] > df['first vote'].median()) & (df['place'] == 1)]
    print(f"\nTrend: High first votes correlate with winning — {len(high_vote_winners)} such cases found.")
    print("\nTrend: No clear correlation between high first votes and winning.")