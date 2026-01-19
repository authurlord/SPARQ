import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Saudi Arabia
saudi_row = df[df['country (or dependent territory)'] == 'saudi arabia']
# Extract the average relative annual growth percentage
avg_growth_saudi = saudi_row['average relative annual growth (%)'].values[0]
print(f"Final Answer: {avg_growth_saudi}")