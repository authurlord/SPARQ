import pandas as pd

df = pd.read_csv('table.csv')
# Filter for primary schools with dcsf number less than 2200
filtered_df = df[(df['type'] == 'primary') & (df['dcsf number'].astype(int) < 2200)]
# Calculate total intake
total_intake = filtered_df['intake'].sum()
print(f"Final Answer: {total_intake}")