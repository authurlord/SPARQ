import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where population density is over 3000 per km square
high_density = df[df['pop density (per km square)'] > 3000]
count_high_density = len(high_density)
print(f"Final Answer: {count_high_density}")