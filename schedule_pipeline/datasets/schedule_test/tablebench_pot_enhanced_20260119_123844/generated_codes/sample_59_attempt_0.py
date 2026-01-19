import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where rank is 4 and gold medals are 4
filtered_row = df[(df['Rank'] == '4') & (df['Gold'] == '4')]
nation = filtered_row['Nation'].iloc[0]
print(f"Final Answer: {nation}")