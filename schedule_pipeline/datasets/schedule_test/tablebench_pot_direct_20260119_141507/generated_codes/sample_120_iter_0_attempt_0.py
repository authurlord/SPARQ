import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out the 'Viewers' column and remove the last row (2014 has no viewers)
viewers_data = df[df['Viewers'].notna()].set_index('Year')['Viewers'].astype(float)

# Create a waterfall chart (cumulative viewership trend)
years = viewers_data.index.tolist()
viewers = viewers_data.values.tolist()

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(years, viewers, color='skyblue', edgecolor='black')
plt.title('Trend in Viewership of the Program')
plt.xlabel('Year')
plt.ylabel('Viewers (in millions)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(years)
plt.tight_layout()
plt.show()