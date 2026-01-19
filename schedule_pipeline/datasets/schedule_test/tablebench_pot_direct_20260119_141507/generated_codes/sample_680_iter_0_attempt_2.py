import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where county is not "united states" and not "new mexico"
filtered_df = df[(df['county'] != 'united states') & (df['county'] != 'new mexico')]

# Extract the 'median household income' and 'population' columns
income = filtered_df['median household income'].astype(float)
population = filtered_df['population'].astype(float)

# Calculate correlation coefficient
correlation = income.corr(population)
print(f"Final Answer: {correlation:.3f}")