import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv('table.csv')

# Parse the 'Years as tallest' column to extract start and end years
def parse_years(year_str):
    # Handle formats like "1882–1886" or "1895–1929"
    match = re.search(r'(\d{4})–(\d{4})', year_str)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return (start + end) // 2  # Midpoint as the time point
    else:
        # Fallback for single year or invalid format
        return int(re.search(r'(\d{4})', year_str).group(1))

# Extract years and height (using only the first number in the height string)
df['height_ft'] = df['Height ft (m)'].str.extract(r'(\d+)').astype(float)
df['year'] = df['Years as tallest'].apply(parse_years)

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['height_ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year')
plt.ylabel('Height (feet)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the final answer as requested (though no specific value is asked, the chart is drawn)
Final Answer: chart_drawn