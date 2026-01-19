import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter only sunken ships (excluding 'Damaged')
sunken_ships = df[df['Fate'] == 'Sunk']
# Convert tonnage to numeric, removing commas
sunken_ships['Tonnage (GRT)'] = sunken_ships['Tonnage (GRT)'].str.replace(',', '').astype(int)
# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sunken_ships['Name'], sunken_ships['Tonnage (GRT)'], color='steelblue')
plt.xlabel('Tonnage (GRT)')
plt.title('Tonnage of Sunken Ships')
plt.gca().invert_yaxis()  # Invert so largest tonnage is at top
plt.tight_layout()
plt.show()