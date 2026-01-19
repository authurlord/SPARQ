import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for honda in 125cc class from 1994 to 1998
honda_125cc = df[(df['team'] == 'honda') & (df['class'] == '125cc') & (df['year'].astype(int) >= 1994) & (df['year'].astype(int) <= 1998)]
# Sort by year to ensure chronological order
honda_125cc = honda_125cc.sort_values('year')
# Extract points and calculate annual increase
points = honda_125cc['points'].astype(int).values
annual_increase = [points[i] - points[i-1] for i in range(1, len(points))]
# Calculate average annual increase
avg_increase = sum(annual_increase) / len(annual_increase)
print(f"Final Answer: {avg_increase:.1f}")