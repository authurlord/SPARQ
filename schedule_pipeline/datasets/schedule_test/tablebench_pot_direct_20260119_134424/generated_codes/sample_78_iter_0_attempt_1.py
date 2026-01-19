import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for Saudi Arabia
saudi_arabia_growth = df[df['country (or dependent territory)'] == 'saudi arabia']['average relative annual growth (%)']
print(f"Final Answer: {saudi_arabia_growth.values[0]}")