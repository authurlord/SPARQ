import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for 'honda' team in '125cc' class from 1994 to 1998
filtered_df = df[(df['team'] == 'honda') & (df['class'] == '125cc') & (df['year'].astype(int) >= 1994) & (df['year'].astype(int) <= 1998)]

# Sort by year to ensure chronological order
filtered_df = filtered_df.sort_values('year')

# Extract points and calculate annual increases
points = filtered_df['points'].values
annual_increases = [points[i] - points[i-1] for i in range(1, len(points))]

# Calculate average annual increase
avg_increase = sum(annual_increases) / len(annual_increases)

print(f"Final Answer: {avg_increase:.1f}")