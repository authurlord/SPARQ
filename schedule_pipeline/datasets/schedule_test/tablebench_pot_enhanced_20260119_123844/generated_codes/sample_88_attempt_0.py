import pandas as pd

df = pd.read_csv('table.csv')
# Sort by gold medals in descending order
df_sorted = df.sort_values(by='gold', ascending=False)
# Get the nation at 4th rank (index 3)
fourth_nation = df_sorted.iloc[3]['nation']
print(f"Final Answer: {fourth_nation}")