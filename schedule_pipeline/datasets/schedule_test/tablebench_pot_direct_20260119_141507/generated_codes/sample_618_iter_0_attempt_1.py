import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Remove the 'total' row since it's a summary
df = df[df['county'] != 'total']

# Convert the '1948' column to numeric (it's already numbers, but safe to convert)
df['1948'] = pd.to_numeric(df['1948'], errors='coerce')

# Sort by 1948 population in descending order and take top 5
top_5_1948 = df.nlargest(5, '1948')

# Sum the 1948 population of these top 5 counties
total_population = top_5_1948['1948'].sum()

print(f"Final Answer: {total_population}")