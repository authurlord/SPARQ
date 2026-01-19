import pandas as pd

df = pd.read_csv('table.csv')
# Filter the data for gold = 4 and silver = 3
filtered_df = df[(df['Gold'] == '4') & (df['Silver'] == '3')]
# Get the nation name
nation = filtered_df['Nation'].values[0]
print(f"Final Answer: {nation}")