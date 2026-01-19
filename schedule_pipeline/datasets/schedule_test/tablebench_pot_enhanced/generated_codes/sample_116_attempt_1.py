import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract start and end years from 'Years as tallest'
df[['start_year', 'end_year']] = df['Years as tallest'].str.split('–', expand=True)
df['start_year'] = pd.to_numeric(df['start_year'])
df['end_year'] = pd.to_numeric(df['end_year'])

# Calculate midpoint of each time period
df['mid_year'] = (df['start_year'] + df['end_year']) / 2

# Extract height in feet (numeric)
df['height_ft'] = df['Height ft (m)'].str.extract('(\d+)').astype(float)

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df['mid_year'], df['height_ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year')
plt.ylabel('Height (ft)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: The line chart is displayed.
print("Final Answer: Line chart displayed")