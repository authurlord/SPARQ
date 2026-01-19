import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

# Extract attendance values
attendance = df['Attendance'].astype(int)

# Plot the trend
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], attendance, marker='o', linestyle='-', color='b')
plt.title('Trend in Team Attendance Over Time')
plt.xlabel('Date')
plt.ylabel('Attendance')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()