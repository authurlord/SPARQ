import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Clean the data: extract season from the second column (e.g., '7th (2000-01)' -> '2000-01')
df['Season'] = df['League.1'].str.extract(r'\((\d{4}-\d{2})\)')[0].fillna(df['League.1'].str.extract(r'(\d{4}-\d{2})')[0])

# For rows where there's no clear year (like "Champion", "Runner-up"), use the full text as label
# But we'll keep only the year part for consistency
# We'll use the Season column as x-axis label

# Prepare data for plotting
data = df[['Season', 'Win', 'Lose']].copy()
data['Season'] = data['Season'].astype(str)

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
plt.bar(data['Season'], data['Win'], label='Wins', color='skyblue')
plt.bar(data['Season'], data['Lose'], bottom=data['Win'], label='Losses', color='lightcoral')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.title('Stacked Bar Chart of Wins and Losses by Season')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()