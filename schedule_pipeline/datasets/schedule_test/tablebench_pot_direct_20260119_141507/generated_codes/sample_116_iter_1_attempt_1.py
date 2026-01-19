import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract start year from 'Years as tallest' (e.g., "1882–1886" -> 1882)
def extract_start_year(year_str):
    if isinstance(year_str, str):
        start = year_str.split('–')[0]
        return int(start)
    return None

df['start_year'] = df['Years as tallest'].apply(extract_start_year)

# Extract height (first number in "Height ft (m)")
df['height_ft'] = df['Height ft (m)'].str.extract(r'(\d+)').astype(float)

# Filter out rows where start_year is None
df = df.dropna(subset=['start_year'])

# Sort by start_year to ensure chronological order
df = df.sort_values('start_year')

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(df['start_year'], df['height_ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year')
plt.ylabel('Height (feet)')
plt.grid(True)
plt.xticks(df['start_year'].unique())
plt.tight_layout()
plt.show()

print(f"Final Answer: Line chart plotted successfully")