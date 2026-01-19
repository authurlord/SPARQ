import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric
df['per capita income'] = pd.to_numeric(df['per capita income'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['median household income'] = pd.to_numeric(df['median household income'], errors='coerce')

# Compute correlation with median household income
correlation_population = df['population'].corr(df['median household income'])
correlation_per_capita = df['per capita income'].corr(df['median household income'])

# Determine which has stronger correlation
if abs(correlation_population) > abs(correlation_per_capita):
    final_answer = "population"
else:
    final_answer = "per capita income"

print(f"Final Answer: {final_answer}")