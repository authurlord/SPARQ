import pandas as pd

df = pd.read_csv('table.csv')

# Filter out non-data rows (those with 'State Hluttaws' or 'Regional Hluttaws' in the first column)
data_rows = df[df['State/Region Hluttaws'] != 'State Hluttaws']
data_rows = data_rows[data_rows['State/Region Hluttaws'] != 'Regional Hluttaws']

# Convert 'Military MPs' and 'Total' to numeric
data_rows['Military MPs'] = pd.to_numeric(data_rows['Military MPs'])
data_rows['Total'] = pd.to_numeric(data_rows['Total'])

# Calculate the proportion of Military MPs to Total
data_rows['proportion'] = (data_rows['Military MPs'] / data_rows['Total']) * 100

# Find the row with the highest proportion
max_row = data_rows.loc[data_rows['proportion'].idxmax()]

# Extract the state/region and percentage
state_region = max_row['State/Region Hluttaws']
percentage = max_row['proportion']

print(f"Final Answer: {state_region}, {percentage:.1f}")