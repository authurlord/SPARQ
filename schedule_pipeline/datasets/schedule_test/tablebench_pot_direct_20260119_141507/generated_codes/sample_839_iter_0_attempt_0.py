import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for 'propane'
propane_row = df[df['fuel gas'] == 'propane']
# Extract upper and lower index kcal/nm³
upper_kcal = float(propane_row['upper index kcal / nm 3'].values[0])
lower_kcal = float(propane_row['lower index kcal / nm 3'].values[0])
difference = upper_kcal - lower_kcal
print(f"Final Answer: {difference}")