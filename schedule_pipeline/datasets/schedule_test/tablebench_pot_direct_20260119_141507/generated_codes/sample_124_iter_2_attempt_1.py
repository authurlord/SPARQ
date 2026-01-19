import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter only sunken ships (fate is 'Sunk')
sunken_ships = df[df['Fate'] == 'Sunk']
# Convert tonnage to integer
sunken_ships['Tonnage (GRT)'] = pd.to_numeric(sunken_ships['Tonnage (GRT)'], errors='coerce')
# Drop any rows with invalid tonnage
sunken_ships = sunken_ships.dropna(subset=['Tonnage (GRT)'])
# Sort by tonnage in descending order
sunken_ships = sunken_ships.sort_values(by='Tonnage (GRT)', ascending=False)
# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sunken_ships['Name'], sunken_ships['Tonnage (GRT)'], color='steelblue')
plt.xlabel('Tonnage (GRT)')
plt.title('Tonnage of Sunken Ships')
plt.gca().invert_yaxis()  # Highest tonnage at top
plt.tight_layout()
plt.show()