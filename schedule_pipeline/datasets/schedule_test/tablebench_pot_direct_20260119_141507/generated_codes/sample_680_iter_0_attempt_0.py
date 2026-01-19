import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'united states' row since it's not a county
df_filtered = df[df['county'] != 'united states']

# Extract the 'median household income' and 'population' columns
income = df_filtered['median household income']
population = df_filtered['population']

# Calculate the correlation coefficient
correlation = income.corr(population)
print(f"Final Answer: {correlation:.3f}")