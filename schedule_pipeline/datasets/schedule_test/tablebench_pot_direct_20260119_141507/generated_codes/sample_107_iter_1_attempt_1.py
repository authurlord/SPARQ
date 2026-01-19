import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter data for ATL team from 1997 to 2003
atl_data = df[(df['Team'] == 'ATL') & (df['Year'].between(1997, 2003))]

# Extract the required columns
wins = atl_data['Regular Season Won']
losses = atl_data['Regular Season Lost']
ties = atl_data['Regular Season Ties']
years = atl_data['Year']

# Create a stacked bar chart
plt.figure(figsize=(10, 6))
x_pos = range(len(years))
width = 0.25

# Plot stacked bars
plt.bar(x_pos, wins, width, label='Wins', color='skyblue')
plt.bar([p + width for p in x_pos], losses, width, label='Losses', color='lightcoral')
plt.bar([p + 2*width for p in x_pos], ties, width, label='Ties', color='gold')

# Labels and title
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.title('ATL Regular Season: Wins, Losses, and Draws (1997–2003)')
plt.xticks([p + width for p in x_pos], years, rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()