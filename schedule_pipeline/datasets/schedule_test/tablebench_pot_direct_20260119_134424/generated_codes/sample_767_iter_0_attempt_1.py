import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in 125cc class
honda_125cc = df[(df['class'] == '125cc') & (df['team'] == 'honda')]
# Sort by year
honda_125cc = honda_125cc.sort_values('year')
# Extract points and years for trend analysis
points = honda_125cc['points'].values
years = honda_125cc['year'].values

# Determine trend
if all(points[i] <= points[i+1] for i in range(len(points)-1)):
    trend = "increasing"
elif all(points[i] >= points[i+1] for i in range(len(points)-1)):
    trend = "decreasing"
else:
    trend = "fluctuating"

print(f"Final Answer: {trend}")