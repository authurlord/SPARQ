import pandas as pd

df = pd.read_csv('table.csv')

# Extract numeric columns for correlation
candidates = df['of candidates nominated'].astype(float)
seats_won = df['of seats won'].astype(float)
total_votes = df['of total votes'].astype(float)

# Compute correlation with seats won
corr_candidates = candidates.corr(seats_won)
corr_votes = total_votes.corr(seats_won)

# Determine which has greater causal impact (higher absolute correlation)
if abs(corr_candidates) > abs(corr_votes):
    final_answer = "number of candidates nominated"
else:
    final_answer = "total number of votes received"

print(f"Final Answer: {final_answer}")