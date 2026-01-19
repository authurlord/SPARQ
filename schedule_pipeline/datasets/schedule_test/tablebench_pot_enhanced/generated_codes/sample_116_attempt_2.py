import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Parse the start year from 'Years as tallest'
df['start_year'] = df['Years as tallest'].str.split('–').str[0].astype(int)

# Extract the height in feet (numeric value before parentheses)
df['Height ft'] = df['Height ft (m)'].str.extract(r'(\d+)').astype(int)

# Sort by start year for proper plotting
df = df.sort_values('start_year')

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df['start_year'], df['Height ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year')
plt.ylabel('Height (ft)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer is not required since the task is to draw a chart