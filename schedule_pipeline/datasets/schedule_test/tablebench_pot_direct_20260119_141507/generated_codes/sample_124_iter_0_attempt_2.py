import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter only sunken ships
sunken_ships = df[df['Fate'] == 'Sunk']
# Convert tonnage to numeric
tonnage = pd.to_numeric(sunken_ships['Tonnage (GRT)'], errors='coerce')
# Drop any invalid entries
tonnage = tonnage.dropna()
# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sunken_ships['Name'], tonnage, color='skyblue')
plt.xlabel('Tonnage (GRT)')
plt.title('Tonnage of Sunken Ships')
plt.gca().invert_yaxis()  # Invert to show largest at top
plt.tight_layout()
plt.show()