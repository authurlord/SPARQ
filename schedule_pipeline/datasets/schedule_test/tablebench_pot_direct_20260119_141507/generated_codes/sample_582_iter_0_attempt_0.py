import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Percentage' column to numeric
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Calculate the difference in percentage from one year to the next
df['diff'] = df['Percentage'].diff()

# Find the year with the largest negative difference (maximum decrease)
max_decrease_row = df[df['diff'] == df['diff'].min()]
year_with_max_decrease = max_decrease_row.iloc[0]['year']

print(f"Final Answer: {year_with_max_decrease}")