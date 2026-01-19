import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in the 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]
# Sort by year to ensure chronological order
honda_125cc = honda_125cc.sort_values('year')
# Extract points and years
points = honda_125cc['points'].values
years = honda_125cc['year'].values

# Analyze trend
trend = "increasing" if points[-1] > points[0] else "decreasing" if points[-1] < points[0] else "fluctuating"
print(f"Final Answer: {trend}")