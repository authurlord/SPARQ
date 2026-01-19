import pandas as pd

df = pd.read_csv('table.csv')
# Convert percentage column to float
df['Percentage (based on registration by the church itself)'] = df['Percentage (based on registration by the church itself)'].astype(float)
# Calculate the difference between consecutive years
df['percentage_change'] = df['Percentage (based on registration by the church itself)'].diff()
# Find the row with the minimum (most negative) change
min_change_row = df[df['percentage_change'] == df['percentage_change'].min()]
# Extract the year
year_of_max_decrease = min_change_row['year'].values[0]
print(f"Final Answer: {year_of_max_decrease}")