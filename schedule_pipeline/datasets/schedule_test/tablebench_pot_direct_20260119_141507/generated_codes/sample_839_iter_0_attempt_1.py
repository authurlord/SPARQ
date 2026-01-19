import pandas as pd

df = pd.read_csv('table.csv')
# Find the row where 'fuel gas' is 'propane'
propane_row = df[df['fuel gas'] == 'propane']
# Extract upper and lower index kcal/nm³ for propane
upper_kcal = float(propane_row['upper index kcal / nm 3'].values[0])
lower_kcal = float(propane_row['lower index kcal / nm 3'].values[0])
# Calculate the difference
difference = upper_kcal - lower_kcal
print(f"Final Answer: {difference}")