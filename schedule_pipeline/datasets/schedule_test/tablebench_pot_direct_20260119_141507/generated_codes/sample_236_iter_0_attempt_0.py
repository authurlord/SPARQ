import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
candidates_nominated = df['of candidates nominated']
seats_won = df['of seats won']
total_votes = df['of total votes']

# Calculate correlation between candidates nominated and seats won
corr_candidates_seats = candidates_nominated.corr(seats_won)

# Calculate correlation between total votes and seats won
corr_votes_seats = total_votes.corr(seats_won)

print(f"Correlation between candidates nominated and seats won: {corr_candidates_seats:.3f}")
print(f"Correlation between total votes and seats won: {corr_votes_seats:.3f}")

# Determine which has a greater absolute correlation
if abs(corr_candidates_seats) > abs(corr_votes_seats):
    final_answer = "number of candidates nominated"
else:
    final_answer = "total number of votes received"

print(f"Final Answer: {final_answer}")