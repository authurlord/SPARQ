import pandas as pd

df = pd.read_csv('table.csv')
# Select only the score columns (excluding 'artist', 'total', and 'points')
score_columns = df.columns[1:-1]  # Skip first (artist) and last two (total, points)
std_devs = df[score_columns].std()
# Find the judge with the highest standard deviation
max_std_judge = std_devs.idxmax()
print(f"Final Answer: {max_std_judge}")