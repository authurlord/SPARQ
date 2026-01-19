import pandas as pd

df = pd.read_csv('table.csv')
# Filter for nations with Total = 8 and Gold = 3
filtered_nation = df[(df['Total'] == 8) & (df['Gold'] == 3)]
# Extract the nation name
nation = filtered_nation['Nation'].iloc[0]
print(f"Final Answer: {nation}")