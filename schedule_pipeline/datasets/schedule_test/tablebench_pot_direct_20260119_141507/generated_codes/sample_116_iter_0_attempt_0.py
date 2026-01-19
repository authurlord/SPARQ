import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv('table.csv')

# Parse 'Years as tallest' to extract start and end years, then compute midpoint
def extract_midpoint(year_str):
    match = re.search(r'(\d{4})–(\d{4})', year_str)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return (start + end) / 2
    else:
        # Fallback if no match
        return pd.NA

df['midpoint'] = df['Years as tallest'].apply(extract_midpoint)

# Convert height to numeric (remove parentheses and extract number)
def parse_height(height_str):
    match = re.search(r'(\d+)', height_str)
    if match:
        return int(match.group(1))
    return 0

df['height_ft'] = df['Height ft (m)'].apply(parse_height)

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(df['midpoint'], df['height_ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year (Midpoint)')
plt.ylabel('Height (feet)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Answer: The line chart has been generated as requested.
Final Answer: chart