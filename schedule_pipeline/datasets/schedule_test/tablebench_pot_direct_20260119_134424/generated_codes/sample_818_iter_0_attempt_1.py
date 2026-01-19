import pandas as pd

df = pd.read_csv('table.csv')
# Drop the header rows and the last total row
df = df[df['State/Region Hluttaws'] != 'State Hluttaws']
df = df[df['State/Region Hluttaws'] != 'Regional Hluttaws']
df = df[df['State/Region Hluttaws'] != '-']

# Convert columns to numeric
df['Military MPs'] = pd.to_numeric(df['Military MPs'])
df['Total'] = pd.to_numeric(df['Total'])

# Calculate proportion of Military MPs to Total
df['proportion'] = df['Military MPs'] / df['Total']

# Find the row with the highest proportion
max_row = df.loc[df['proportion'].idxmax()]

# Extract state/region and percentage
state_region = max_row['State/Region Hluttaws']
percentage = max_row['proportion'] * 100

print(f"Final Answer: {state_region}, {percentage:.1f}")