import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the 'Viewers' column: remove ' million' and convert to float
df['Viewers'] = df['Viewers'].str.replace(' million', '').astype(float)

# Drop the row where viewership is missing (2014)
df = df.dropna(subset=['Viewers'])

# Extract years and viewership
years = df['Year'].astype(int)
viewership = df['Viewers']

# Create a waterfall-style trend chart (using line plot for trend)
plt.figure(figsize=(10, 6))
plt.plot(years, viewership, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)
plt.title('Trend in Viewership of the Program')
plt.xlabel('Year')
plt.ylabel('Viewers (in millions)')
plt.grid(True)
plt.xticks(years)
plt.tight_layout()

# Show the plot
plt.show()