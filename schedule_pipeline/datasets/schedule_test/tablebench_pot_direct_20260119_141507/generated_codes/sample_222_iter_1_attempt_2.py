import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header row)
df = df.iloc[1:]

# Convert all relevant columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Extract the columns we care about
latvian_col = '95 - 96'
russian_col = '99 - 00'
total_col = 'total'

# Select only the relevant years
years = df.columns[1:-1]  # Exclude first and last (school year)

# Extract values for latvian, russian, and total
latvian_data = df[latvian_col]
russian_data = df['russian']
total_data = df[total_col]

# Compute proportions
proportion_latvian = latvian_data / total_data
proportion_russian = russian_data / total_data

# Compute the change in proportion over time (differences between consecutive years)
delta_latvian = proportion_latvian.diff()
delta_russian = proportion_russian.diff()

# Compute average rate of change (mean of differences)
avg_change_latvian = delta_latvian.mean()
avg_change_russian = delta_russian.mean()

# Determine which one has a larger absolute change
abs_latvian_change = abs(avg_change_latvian)
abs_russian_change = abs(avg_change_russian)

# Final answer: Which proportion changed more significantly?
if abs_latvian_change > abs_russian_change:
    result = "Latvian"
else:
    result = "Russian"

print(f"Final Answer: {result}")