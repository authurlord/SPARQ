import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'total' row
data = df[df['year'] != 'total'].copy()
data['year'] = data['year'].astype(int)
data['tries'] = data['tries'].astype(int)
data['points'] = data['points'].astype(int)

# Calculate the linear trend (slope) for tries and points
years = data['year']
tries = data['tries']
points = data['points']

# Linear regression: y = mx + b
# Using numpy for slope calculation
m_tries = (len(years) * (years * tries).sum() - years.sum() * tries.sum()) / \
          (len(years) * (years**2).sum() - years.sum()**2)
m_points = (len(years) * (years * points).sum() - years.sum() * points.sum()) / \
           (len(years) * (years**2).sum() - years.sum()**2)

# Project for next 5 years (2011 to 2015)
future_years = range(2011, 2016)
projected_tries = [m_tries * y + (tries.iloc[0] - m_tries * years.iloc[0]) for y in future_years]
projected_points = [m_points * y + (points.iloc[0] - m_points * years.iloc[0]) for y in future_years]

# Average projected values over next 5 years
avg_projected_tries = sum(projected_tries) / len(projected_tries)
avg_projected_points = sum(projected_points) / len(projected_points)

print(f"Final Answer: {avg_projected_tries:.1f}, {avg_projected_points:.1f}")