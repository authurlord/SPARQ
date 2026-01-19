import pandas as pd

df = pd.read_csv('table.csv')
# Filter for the 3000 m steeplechase event with 1st position
filtered_df = df[(df['Event'] == '3000 m steeplechase') & (df['Position'] == '1st')]
# Get the competition from the filtered row
competition = filtered_df['Competition'].values[0]
print(f"Final Answer: {competition}")