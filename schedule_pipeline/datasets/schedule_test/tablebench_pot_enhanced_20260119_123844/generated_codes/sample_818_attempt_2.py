import pandas as pd

df = pd.read_csv('table.csv')

# Drop the first two rows which are headers
df = df.iloc[2:]

# Convert relevant columns to numeric
df['Military MPs'] = pd.to_numeric(df['Military MPs'])
df['Total'] = pd.to_numeric(df['Total'])

# Calculate the proportion of Military MPs to Total
df['proportion'] = df['Military MPs'] / df['Total']

# Find the row with the maximum proportion
max_row = df.loc[df['proportion'].idxmax()]

# Extract the state/region and calculate percentage
state_region = max_row['State/Region Hluttaws']
percentage = (max_row['Military MPs'] / max_row['Total']) * 100

print(f"Final Answer: {state_region}, {percentage:.1f}%")