import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where event is '3000 m steeplechase' and position is '1st'
filtered_df = df[(df['Event'] == '3000 m steeplechase') & (df['Position'] == '1st')]
# Get the competition from the filtered row
competition = filtered_df['Competition'].iloc[0]
print(f"Final Answer: {competition}")