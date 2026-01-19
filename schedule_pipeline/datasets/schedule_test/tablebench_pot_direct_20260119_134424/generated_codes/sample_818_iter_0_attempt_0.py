import pandas as pd

df = pd.read_csv('table.csv')

# Filter out the header rows and the total row
df_filtered = df[df['State/Region Hluttaws'] != 'State Hluttaws']
df_filtered = df_filtered[df_filtered['State/Region Hluttaws'] != 'Regional Hluttaws']
df_filtered = df_filtered[df_filtered['State/Region Hluttaws'] != '-']

# Convert 'Military MPs' and 'Total' columns to numeric
df_filtered['Military MPs'] = pd.to_numeric(df_filtered['Military MPs'])
df_filtered['Total'] = pd.to_numeric(df_filtered['Total'])

# Calculate the proportion of Military MPs to Total
df_filtered['proportion'] = df_filtered['Military MPs'] / df_filtered['Total']

# Find the row with the highest proportion
max_row = df_filtered.loc[df_filtered['proportion'].idxmax()]

# Extract the state/region name and calculate percentage
state_region = max_row['State/Region Hluttaws']
percentage = max_row['proportion'] * 100

print(f"Final Answer: {state_region}, {percentage:.1f}")