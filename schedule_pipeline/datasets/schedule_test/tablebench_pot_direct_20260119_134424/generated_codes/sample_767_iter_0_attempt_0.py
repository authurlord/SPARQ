import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]
# Sort by year
honda_125cc = honda_125cc.sort_values('year')
# Extract points and years for trend analysis
points = honda_125cc['points'].tolist()
years = honda_125cc['year'].tolist()

# Determine trend
trend = "increasing" if all(points[i] <= points[i+1] for i in range(len(points)-1)) else "not consistently increasing"
print(f"Final Answer: {trend}")