import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the start year from 'Years as tallest'
df['Start Year'] = df['Years as tallest'].str.split('–').str[0].astype(int)

# Extract height in feet (numeric value before parentheses)
df['Height ft'] = df['Height ft (m)'].str.extract(r'(\d+)').astype(int)

# Sort by start year for proper line chart
df = df.sort_values('Start Year')

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df['Start Year'], df['Height ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year')
plt.ylabel('Height (ft)')
plt.grid(True)
plt.tight_layout()
plt.show()

print("Final Answer: Line chart displayed")