import pandas as pd

df = pd.read_csv('table.csv')
# Filter nation with 4 gold and 3 silver medals
result = df[(df['Gold'] == 4) & (df['Silver'] == 3)]['Nation'].values
print(f"Final Answer: {result[0]}")