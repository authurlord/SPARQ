import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Remove the 'total' row and convert the 1948 column to numeric
df_filtered = df[df['county'] != 'total']
df_filtered['1948'] = pd.to_numeric(df_filtered['1948'], errors='coerce')

# Sort by 1948 population in descending order and take top 5
top_5_1948 = df_filtered.sort_values(by='1948', ascending=False).head(5)

# Calculate the sum of the top 5 counties' 1948 populations
total_population_top_5 = top_5_1948['1948'].sum()

print(f"Final Answer: {total_population_top_5}")