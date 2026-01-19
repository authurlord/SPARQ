import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Filter for ATL team and years 1997 to 2003
atl_data = df[(df['Team'] == 'ATL') & (df['Year'].astype(str).str.startswith('1997')) | 
              (df['Year'].astype(str).str.startswith('1998')) | 
              (df['Year'].astype(str).str.startswith('1999')) | 
              (df['Year'].astype(str).str.startswith('2000')) | 
              (df['Year'].astype(str).str.startswith('2001')) | 
              (df['Year'].astype(str).str.startswith('2002')) | 
              (df['Year'].astype(str).str.startswith('2003'))]

# Drop the "ATL Total" row which is a summary
atl_data = atl_data[atl_data['Year'] != 'ATL Total']

# Convert Year to integer for proper sorting
atl_data['Year'] = atl_data['Year'].astype(int)

# Select relevant columns
data = atl_data[['Year', 'Regular Season Won', 'Regular Season Lost', 'Regular Season Ties']]

# Prepare for plotting
years = data['Year']
wins = data['Regular Season Won']
losses = data['Regular Season Lost']
ties = data['Regular Season Ties']

# Create a stacked bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.25
index = range(len(years))

plt.bar(index, wins, bar_width, label='Wins', color='skyblue')
plt.bar([i + bar_width for i in index], losses, bar_width, label='Losses', color='salmon')
plt.bar([i + 2*bar_width for i in index], ties, bar_width, label='Ties', color='lightgreen')

plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.title('ATL Team Regular Season: Wins, Losses, and Ties (1997–2003)')
plt.xticks([i + bar_width for i in index], years)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Final Answer: The chart has been generated as requested.
Final Answer: chart_generated