import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert Attendance from string to integer (remove commas)
df['Attendance'] = df['Attendance'].str.replace(',', '').astype(int)

# Sort by date
df = df.sort_values('Date')

# Create a waterfall chart (step plot)
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Attendance'], marker='o', linestyle='-', color='blue', label='Attendance')

# Add labels and title
plt.title('Trend in Team Attendance Over Time')
plt.xlabel('Date')
plt.ylabel('Attendance')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()