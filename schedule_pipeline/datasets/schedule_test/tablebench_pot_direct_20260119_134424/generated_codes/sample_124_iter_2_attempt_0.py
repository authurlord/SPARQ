import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter for sunken ships
sunken_ships = df[df['Fate'] == 'Sunk']
# Convert tonnage to numeric, removing commas
sunken_ships['Tonnage (GRT)'] = sunken_ships['Tonnage (GRT)'].str.replace(',', '').astype(int)
# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sunken_ships['Name'], sunken_ships['Tonnage (GRT)'], color='skyblue')
plt.xlabel('Tonnage (GRT)')
plt.ylabel('Ship Name')
plt.title('Tonnage of Sunken Ships')
plt.gca().invert_yaxis()  # To show the highest tonnage at the top
plt.tight_layout()
plt.show()