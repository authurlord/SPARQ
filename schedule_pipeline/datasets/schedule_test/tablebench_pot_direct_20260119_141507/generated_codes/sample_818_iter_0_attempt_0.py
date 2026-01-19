import pandas as pd

df = pd.read_csv('table.csv')

# Drop the first row (headers) and the last row (summary)
df_clean = df.drop([0, -1]).copy()

# Remove rows where 'Total' is NaN or 0
df_clean = df_clean[df_clean['Total'] != 0]

# Calculate proportion of Military MPs to Total MPs
df_clean['proportion'] = df_clean['Military MPs'] / df_clean['Total']

# Find the row with maximum proportion
max_row = df_clean.loc[df_clean['proportion'].idxmax()]

# Extract state/region name and percentage
state_region = max_row['State/Region Hluttaws']
percentage = max_row['proportion'] * 100

print(f"Final Answer: {state_region}, {percentage:.1f}")