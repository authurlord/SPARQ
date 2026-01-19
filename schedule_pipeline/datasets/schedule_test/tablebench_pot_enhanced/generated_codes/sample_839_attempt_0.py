import pandas as pd

df = pd.read_csv('table.csv')
# Filter for propane
propane_row = df[df['fuel gas'] == 'propane']
# Extract upper and lower indices
upper_index = float(propane_row['upper index kcal / nm 3'].values[0])
lower_index = float(propane_row['lower index kcal / nm 3'].values[0])
# Calculate difference
difference = upper_index - lower_index
print(f"Final Answer: {difference}")