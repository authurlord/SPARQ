import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter out the 'Career Totals' row
df_filtered = df[df['Team'] != 'Career Totals']

# Convert 'Year' to integer for proper plotting
df_filtered['Year'] = pd.to_numeric(df_filtered['Year'], errors='coerce')
df_filtered = df_filtered.dropna(subset=['Year'])

# Extract year and attacks
years = df_filtered['Year']
attacks = df_filtered['Att']

# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(years, attacks, marker='o', linestyle='-', color='b')
plt.title('Trend in Number of Attacks Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Attacks (Att)')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()