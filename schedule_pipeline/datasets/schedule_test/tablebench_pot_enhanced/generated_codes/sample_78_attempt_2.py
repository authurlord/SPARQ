import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row where country is 'saudi arabia'
saudi_arabia_growth = df[df['country (or dependent territory)'] == 'saudi arabia']['average relative annual growth (%)']
# Extract the value and print
print(f"Final Answer: {saudi_arabia_growth.values[0]}")