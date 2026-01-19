import pandas as pd

df = pd.read_csv('table.csv')
# Filter for total medals = 8 and gold medals = 3
result = df[(df['Total'] == 8) & (df['Gold'] == 3)]
# Extract the nation name
nation = result['Nation'].iloc[0]
print(f"Final Answer: {nation}")