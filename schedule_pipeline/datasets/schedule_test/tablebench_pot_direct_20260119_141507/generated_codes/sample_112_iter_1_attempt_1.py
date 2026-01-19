import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter out the "Career Totals" row
df_filtered = df[df['Year'] != 'Career Totals']

# Convert 'Year' to integer for proper plotting
df_filtered['Year'] = pd.to_numeric(df_filtered['Year'], errors='coerce')
df_filtered = df_filtered.dropna(subset=['Year'])

# Extract year and attacks
years = df_filtered['Year']
attacks = df_filtered['Att']

# Create the line chart
plt.figure(figsize=(10, 6))
plt.plot(years, attacks, marker='o', linestyle='-', color='b')
plt.title('Trend in the Athlete\'s Number of Attacks')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.grid(True)
plt.xticks(years[::1])  # Show every year
plt.tight_layout()
plt.show()

# Final Answer: The line chart has been plotted successfully.
Final Answer: chart