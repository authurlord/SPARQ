import pandas as pd

df = pd.read_csv('table.csv')
# Select only the score columns (excluding 'artist', 'total', 'points')
score_columns = df.columns[1:-2]  # Columns from 'jaanu någisto' to 'siim nestor'
std_devs = df[score_columns].std()
# Find the judge with the highest standard deviation
most_varied_judge = std_devs.idxmax()
print(f"Final Answer: {most_varied_judge}")