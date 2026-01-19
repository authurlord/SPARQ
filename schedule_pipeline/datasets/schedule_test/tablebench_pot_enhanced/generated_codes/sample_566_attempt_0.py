import pandas as pd

df = pd.read_csv('table.csv')
# Convert Year to integer for proper sorting
df['Year'] = pd.to_numeric(df['Year'])
# Get the earliest two years
earliest_years = df['Year'].nsmallest(2).unique()
# Filter data for the earliest two years
filtered_df = df[df['Year'].isin(earliest_years)]
# Calculate total quantity
total_quantity = filtered_df['Quantity'].sum()
print(f"Final Answer: {total_quantity}")