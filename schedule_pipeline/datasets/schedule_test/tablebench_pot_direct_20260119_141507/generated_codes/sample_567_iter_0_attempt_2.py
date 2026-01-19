import pandas as pd

df = pd.read_csv('table.csv')
# Extract issue price for 2000 and 2010
issue_2000 = df[df['year'] == '2000']['issue price'].values[0]
issue_2010 = df[df['year'] == '2010']['issue price'].values[0]

# Calculate percentage increase
percentage_increase = ((issue_2010 - issue_2000) / issue_2000) * 100
print(f"Final Answer: {percentage_increase:.2f}")