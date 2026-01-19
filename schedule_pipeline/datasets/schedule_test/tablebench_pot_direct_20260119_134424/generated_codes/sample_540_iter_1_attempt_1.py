import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for 'honda' team and '125cc' class
filtered_df = df[(df['team'] == 'honda') & (df['class'] == '125cc')]

# Sort by year to ensure chronological order
filtered_df = filtered_df.sort_values(by='year')

# Convert year to integer and extract points
points = filtered_df['points'].astype(int)
years = filtered_df['year'].astype(int)

# Calculate annual increases in points
annual_increases = []
for i in range(1, len(points)):
    increase = points.iloc[i] - points.iloc[i-1]
    annual_increases.append(increase)

# Calculate average annual increase
avg_increase = sum(annual_increases) / len(annual_increases)

print(f"Final Answer: {avg_increase:.1f}")