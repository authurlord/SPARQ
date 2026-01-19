import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert Date to datetime and sort by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Extract attendance values
attendance = df['Attendance'].astype(int)
dates = df['Date']

# Create a waterfall-style trend chart (line plot with markers)
plt.figure(figsize=(12, 6))
plt.plot(dates, attendance, marker='o', linestyle='-', color='blue', linewidth=2, markersize=6)
plt.title('Trend in Team Attendance Over Time')
plt.xlabel('Date')
plt.ylabel('Attendance')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()