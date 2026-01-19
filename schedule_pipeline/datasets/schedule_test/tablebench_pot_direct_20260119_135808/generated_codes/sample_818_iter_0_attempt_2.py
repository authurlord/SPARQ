import pandas as pd

df = pd.read_csv('table.csv')
# Remove the header rows and the last summary row
data_rows = df.iloc[1:-1]
# Convert 'Military MPs' and 'Total' columns to numeric
data_rows['Military MPs'] = pd.to_numeric(data_rows['Military MPs'])
data_rows['Total'] = pd.to_numeric(data_rows['Total'])
# Calculate proportion of Military MPs to Total
data_rows['proportion'] = data_rows['Military MPs'] / data_rows['Total']
# Find the row with the highest proportion
max_row = data_rows.loc[data_rows['proportion'].idxmax()]
# Get the state/region name and calculate percentage
state_region = max_row['State/Region Hluttaws']
percentage = max_row['proportion'] * 100
print(f"Final Answer: {state_region}, {percentage:.1f}%")