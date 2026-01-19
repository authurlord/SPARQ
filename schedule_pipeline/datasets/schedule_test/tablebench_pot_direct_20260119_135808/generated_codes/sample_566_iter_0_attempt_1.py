import pandas as pd

df = pd.read_csv('table.csv')
# Sort by Year to ensure earliest years are correctly identified
df_sorted = df.sort_values('Year')
# Select the first two years (1893 and 1894)
earliest_two_years = df_sorted['Year'].unique()[:2]
# Filter data for the earliest two years
filtered_data = df_sorted[df_sorted['Year'].isin(earliest_two_years)]
# Sum the Quantity column
total_quantity = filtered_data['Quantity'].sum()
print(f"Final Answer: {total_quantity}")