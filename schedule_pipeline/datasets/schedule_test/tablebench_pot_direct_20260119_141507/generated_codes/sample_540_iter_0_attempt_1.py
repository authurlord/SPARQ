import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for 'honda' team in '125cc' class
filtered_df = df[(df['team'] == 'honda') & (df['class'] == '125cc')]

# Extract points and years
points = filtered_df['points'].astype(int)
years = pd.to_numeric(filtered_df['year'])

# Calculate the difference in points between consecutive years
point_diffs = []
for i in range(1, len(years)):
    diff = points.iloc[i] - points.iloc[i-1]
    point_diffs.append(diff)

# Compute average annual increase
average_increase = sum(point_diffs) / len(point_diffs) if point_diffs else 0
print(f"Final Answer: {average_increase:.1f}")