import pandas as pd

df = pd.read_csv('table.csv')
# Ensure 'against' is numeric and compute mean and standard deviation
against_values = pd.to_numeric(df['against'], errors='coerce')
mean_against = against_values.mean()
std_against = against_values.std()
print(f"Final Answer: {mean_against:.1f}, {std_against:.1f}")