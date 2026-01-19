import pandas as pd

df = pd.read_csv('table.csv')
# Filter for position 37th in the 20 km walk event
filtered_row = df[(df['Position'] == '37th') & (df['Event'] == '20 km walk')]
competition = filtered_row['Competition'].iloc[0]
print(f"Final Answer: {competition}")