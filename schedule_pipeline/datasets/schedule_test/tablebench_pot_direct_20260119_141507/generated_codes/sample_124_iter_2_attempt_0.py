import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter only sunken ships (exclude 'Damaged')
sunk_ships = df[df['Fate'] == 'Sunk']
# Convert tonnage to numeric
sunk_ships['Tonnage (GRT)'] = pd.to_numeric(sunk_ships['Tonnage (GRT)'], errors='coerce')
# Drop any rows with invalid tonnage
sunk_ships = sunk_ships.dropna(subset=['Tonnage (GRT)'])

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sunk_ships['Name'], sunk_ships['Tonnage (GRT)'], color='skyblue')
plt.xlabel('Tonnage (GRT)')
plt.title('Tonnage of Sunken Ships')
plt.gca().invert_yaxis()  # Invert so highest tonnage is at top
plt.tight_layout()
plt.show()