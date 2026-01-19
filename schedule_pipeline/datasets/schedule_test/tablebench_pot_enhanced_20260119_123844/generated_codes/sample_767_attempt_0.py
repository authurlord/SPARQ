import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for Honda in the 125cc class
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc')]
# Sort by year to ensure chronological order
honda_125cc = honda_125cc.sort_values('year')
# Check the trend in points
points = honda_125cc['points'].tolist()
trend = "increasing" if all(points[i] <= points[i+1] for i in range(len(points)-1)) else "not consistently increasing"
print(f"Final Answer: {trend}")