import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv('table.csv')

# Extract end year from "Years as tallest"
df['end_year'] = df['Years as tallest'].str.extract(r'(\d{4})').astype(int)

# Extract height in feet (remove text like "(m)")
df['height_ft'] = df['Height ft (m)'].str.extract(r'(\d+)').astype(float)

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(df['end_year'], df['height_ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('End Year')
plt.ylabel('Height (feet)')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Final Answer: Line chart plotted successfully")