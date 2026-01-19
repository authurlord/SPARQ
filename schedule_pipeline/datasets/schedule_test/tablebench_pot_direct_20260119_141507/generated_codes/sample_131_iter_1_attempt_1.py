import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Clean the 'League.1' column to extract year
df['year'] = df['League.1'].str.extract(r'\((\d{4})\)', expand=False).fillna(0).astype(int)

# Drop rows where 'Win' or 'Lose' are not numeric (though they are)
# Convert 'Win' and 'Lose' to integers
df['Win'] = pd.to_numeric(df['Win'], errors='coerce')
df['Lose'] = pd.to_numeric(df['Lose'], errors='coerce')

# Drop rows with NaN in Win or Lose
df = df.dropna(subset=['Win', 'Lose'])

# Group by year and aggregate wins and losses
yearly_stats = df.groupby('year')[['Win', 'Lose']].sum().reset_index()

# Create stacked bar chart
plt.figure(figsize=(10, 6))
plt.bar(yearly_stats['year'], yearly_stats['Win'], label='Wins', color='skyblue')
plt.bar(yearly_stats['year'], yearly_stats['Lose'], bottom=yearly_stats['Win'], label='Losses', color='lightcoral')
plt.xlabel('Year')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins and Losses by Year')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print summary for clarity
print(f"Final Answer: Stacked bar chart generated for wins and losses by year")