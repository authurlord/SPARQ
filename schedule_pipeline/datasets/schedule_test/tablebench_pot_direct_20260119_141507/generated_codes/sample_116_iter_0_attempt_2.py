import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract years and height
df['start_year'] = df['Years as tallest'].str.extract(r'(\d{4})-(\d{4})')[0].astype(int)
df['end_year'] = df['Years as tallest'].str.extract(r'(\d{4})-(\d{4})')[1].astype(int)
df['mid_year'] = (df['start_year'] + df['end_year']) / 2

# Extract height (first number in string like "157 (48)")
df['height_ft'] = df['Height ft (m)'].str.extract(r'(\d+)').astype(float)

# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(df['mid_year'], df['height_ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year')
plt.ylabel('Height (feet)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the data points for clarity (optional)
print(f"Final Answer: Line chart generated successfully")