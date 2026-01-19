import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Date' to datetime and sort by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Convert 'Attendance' to numeric
df['Attendance'] = pd.to_numeric(df['Attendance'], errors='coerce')

# Drop any rows with NaN attendance
df = df.dropna(subset=['Attendance'])

# Plot the attendance trend over time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Attendance'], marker='o', linestyle='-', color='b')
plt.title('Trend in Team Attendance Over Time')
plt.xlabel('Date')
plt.ylabel('Attendance')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()