import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Saudi Arabia
saudi_arabia_row = df[df['country (or dependent territory)'] == 'saudi arabia']
# Extract the average relative annual growth rate
growth_rate = saudi_arabia_row['average relative annual growth (%)'].values[0]
print(f"Final Answer: {growth_rate}")