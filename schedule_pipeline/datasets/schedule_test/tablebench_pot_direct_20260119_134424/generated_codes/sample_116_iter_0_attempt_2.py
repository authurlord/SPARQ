import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the start year from 'Years as tallest'
df['start_year'] = df['Years as tallest'].str.split('–').str[0].astype(int)

# Extract the height in feet (numeric value)
df['height_ft'] = df['Height ft (m)'].str.extract('(\d+)').astype(int)

# Sort by start year for proper line plotting
df = df.sort_values('start_year')

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(df['start_year'], df['height_ft'], marker='o', linestyle='-', color='b')
plt.title('Trend in Maximum Building Height Over Time')
plt.xlabel('Year Building Became Tallest')
plt.ylabel('Height (ft)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: The chart has been displayed.
Final Answer: None