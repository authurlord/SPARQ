import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Total is 8 and Gold is 3
filtered_nation = df[(df['Total'] == 8) & (df['Gold'] == 3)]
# Extract the nation name
nation = filtered_nation['Nation'].iloc[0]
print(f"Final Answer: {nation}")