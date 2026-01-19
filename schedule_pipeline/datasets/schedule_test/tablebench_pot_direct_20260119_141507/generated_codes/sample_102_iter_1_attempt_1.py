import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert Period to a list of strings for x-axis
periods = df['Period']

# Extract live births and deaths
births = df['Live births per year'].astype(float)
deaths = df['Deaths per year'].astype(float)

# Create the area chart
plt.figure(figsize=(12, 6))
plt.fill_between(periods, births, label='Live Births', alpha=0.7)
plt.fill_between(periods, deaths, label='Deaths', alpha=0.7)
plt.title('Number of Live Births and Deaths Over Time Periods')
plt.xlabel('Time Period')
plt.ylabel('Number of Events')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: The area chart has been plotted successfully.
Final Answer: chart_plotted