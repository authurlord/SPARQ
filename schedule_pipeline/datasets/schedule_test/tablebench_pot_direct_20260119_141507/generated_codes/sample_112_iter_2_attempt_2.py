import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter out the "Career Totals" row
df_filtered = df[df['Year'] != 'Career Totals']

# Convert 'Year' to integer for proper plotting
df_filtered['Year'] = pd.to_numeric(df_filtered['Year'], errors='coerce')

# Drop rows with invalid year values
df_filtered = df_filtered.dropna(subset=['Year'])

# Extract the 'Year' and 'Att' columns
years = df_filtered['Year']
attacks = df_filtered['Att']

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(years, attacks, marker='o', linestyle='-', color='b')
plt.title('Trend in the Number of Attacks Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()