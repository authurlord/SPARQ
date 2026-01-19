import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
population = df['population']
per_capita_income = df['per capita income']
median_household_income = df['median household income']

# Calculate correlation with median household income
corr_population = population.corr(median_household_income)
corr_per_capita = per_capita_income.corr(median_household_income)

# Determine which has a stronger correlation (higher absolute value)
if abs(corr_population) > abs(corr_per_capita):
    final_answer = "population"
else:
    final_answer = "per capita income"

print(f"Final Answer: {final_answer}")