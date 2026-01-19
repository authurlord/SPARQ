import pandas as pd

df = pd.read_csv('table.csv')
# Extract relevant rows: 1850 to 1870
relevant_data = df[df['Year'].isin(['1850', '1855', '1860', '1865', '1870'])]
relevant_data = relevant_data.sort_values('Year')

# Convert the '-' column to numeric (remove commas)
relevant_data['value'] = relevant_data['-'].str.replace(',', '').astype(int)

# Calculate the increase between consecutive years
relevant_data['increase'] = relevant_data['value'].diff()

# Find the row with the maximum increase
max_increase_row = relevant_data.loc[relevant_data['increase'].idxmax()]

# The value in the year with the highest increase
final_value = max_increase_row['value']
print(f"Final Answer: {final_value}")