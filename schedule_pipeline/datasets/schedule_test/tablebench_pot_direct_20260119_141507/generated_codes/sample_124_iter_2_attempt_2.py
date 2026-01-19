import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter only sunken ships (including 'Sunk (mine)')
sunken_ships = df[df['Fate'].str.contains('Sunk', case=False, na=False)]
# Convert tonnage to numeric
sunken_ships['Tonnage (GRT)'] = pd.to_numeric(sunken_ships['Tonnage (GRT)'], errors='coerce')
# Drop any NaN values due to conversion errors
sunken_ships = sunken_ships.dropna(subset=['Tonnage (GRT)'])

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sunken_ships['Name'], sunken_ships['Tonnage (GRT)'], color='skyblue')
plt.xlabel('Tonnage (GRT)')
plt.title('Tonnage of Sunken Ships')
plt.gca().invert_yaxis()  # Invert so highest tonnage is at top
plt.tight_layout()
plt.show()