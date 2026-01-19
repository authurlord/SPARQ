import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the first two rows (headers) and the last row (summary)
df_data = df.iloc[1:-1]
# Convert 'Military MPs' and 'Total' columns to numeric
df_data['Military MPs'] = pd.to_numeric(df_data['Military MPs'])
df_data['Total'] = pd.to_numeric(df_data['Total'])
# Calculate the proportion of Military MPs to Total
df_data['proportion'] = df_data['Military MPs'] / df_data['Total']
# Find the row with the highest proportion
max_row = df_data.loc[df_data['proportion'].idxmax()]
# Extract the state/region name and calculate percentage
state_region = max_row['State/Region Hluttaws']
percentage = (max_row['Military MPs'] / max_row['Total']) * 100
print(f"Final Answer: {state_region}, {percentage:.1f}%")