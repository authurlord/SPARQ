import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nations with 5 gold and 2 silver medals
filtered_nation = df[(df['Gold'] == '5') & (df['Silver'] == '2')]
# Extract the nation name
nation = filtered_nation['Nation'].values[0]
print(f"Final Answer: {nation}")