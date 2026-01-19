import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
branches = df['# of Branches As of 30 September 2012']
assets = df['Total Assets (million TL) As of 30 September 2012']

# Calculate the correlation
correlation = branches.corr(assets)
print(f"Final Answer: {correlation:.2f}")