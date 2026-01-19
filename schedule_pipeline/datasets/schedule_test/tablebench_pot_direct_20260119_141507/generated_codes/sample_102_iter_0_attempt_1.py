import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert the 'Period' column to a list of years for x-axis
periods = df['Period']
births = df['Live births per year'].str.replace(' ', '').astype(int)
deaths = df['Deaths per year'].str.replace(' ', '').astype(int)

# Create a DataFrame for plotting
data = pd.DataFrame({
    'Period': periods,
    'Live births': births,
    'Deaths': deaths
})

# Plot area chart
plt.figure(figsize=(12, 6))
plt.stackplot(data['Period'], data['Live births'], data['Deaths'], labels=['Live births', 'Deaths'])
plt.title('Number of Births and Deaths Over Time Periods')
plt.xlabel('Time Period')
plt.ylabel('Number of Events')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()