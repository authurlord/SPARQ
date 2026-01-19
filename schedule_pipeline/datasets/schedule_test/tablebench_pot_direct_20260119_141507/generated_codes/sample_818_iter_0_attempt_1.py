import pandas as pd

df = pd.read_csv('table.csv')
# Drop the first row (headers) and the last row (summary)
# Also drop rows where 'State/Region Hluttaws' is '-' or empty
df_filtered = df.drop(df.index[0])  # Remove first row (headers)
df_filtered = df_filtered[df_filtered['State/Region Hluttaws'] != '-']

# Convert relevant columns to numeric
df_filtered['Military MPs'] = pd.to_numeric(df_filtered['Military MPs'], errors='coerce')
df_filtered['Total'] = pd.to_numeric(df_filtered['Total'], errors='coerce')

# Calculate proportion of Military MPs to Total MPs
df_filtered['proportion'] = df_filtered['Military MPs'] / df_filtered['Total']

# Find the row with the highest proportion
max_row = df_filtered.loc[df_filtered['proportion'].idxmax()]
state_region = max_row['State/Region Hluttaws']
percentage = max_row['proportion'] * 100

print(f"Final Answer: {state_region}, {percentage:.1f}")