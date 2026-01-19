import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the median of 'pop density (per km square)'
median_density = df['pop density (per km square)'].median()
print(f"Final Answer: {median_density:.2f}")