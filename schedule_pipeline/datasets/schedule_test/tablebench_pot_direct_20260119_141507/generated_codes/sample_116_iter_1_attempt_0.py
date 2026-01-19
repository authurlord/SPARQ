import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv('table.csv')

# Extract years from "Years as tallest" column
def parse_years(year_str):
    match = re.search(r'(\d{4})–(\d{4})', year_str)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return (start + end) // 2  # Midpoint year
    else:
        # Fallback: use the year if no range
        return int(year_str.split('-')[0])

# Apply parsing to create a new column for years
df['year'] = df['Years as tallest'].apply(parse_years)

# Extract height (first number in string like '157 (48)')
df['height_ft'] = df['Height ft (m)'].str.extract(r'(\d+)').astype(float)

# Drop rows with missing height
df = df.dropna(subset=['height_ft'])

# Sort by year to ensure chronological order
df = df.sort_values(by='year')

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['height_ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year')
plt.ylabel('Height (feet)')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Final Answer: Line chart plotted successfully")