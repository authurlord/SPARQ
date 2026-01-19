import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nation with 4 gold and 3 silver medals
result = df[(df['Gold'] == '4') & (df['Silver'] == '3')]
# Extract the nation name
nation = result['Nation'].values[0]
print(f"Final Answer: {nation}")