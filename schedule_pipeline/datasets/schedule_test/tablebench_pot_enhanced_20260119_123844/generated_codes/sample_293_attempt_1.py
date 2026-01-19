import pandas as pd

df = pd.read_csv('table.csv')
# Filter for position 37th in 20 km walk
filtered_row = df[(df['Position'] == '37th') & (df['Event'] == '20 km walk')]
# Extract the competition name
competition = filtered_row['Competition'].values[0]
print(f"Final Answer: {competition}")