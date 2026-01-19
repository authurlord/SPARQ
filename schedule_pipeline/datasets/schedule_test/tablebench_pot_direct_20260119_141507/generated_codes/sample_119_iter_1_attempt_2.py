import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Date' to datetime for proper sorting
df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y')

# Convert 'Attendance' to integer (remove commas)
df['Attendance'] = df['Attendance'].str.replace(',', '').astype(int)

# Sort by date to ensure chronological order
df = df.sort_values(by='Date')

# Extract date and attendance for plotting
dates = df['Date']
attendance = df['Attendance']

# Create a waterfall chart (cumulative trend)
plt.figure(figsize=(12, 6))
plt.plot(dates, attendance, marker='o', linestyle='-', color='blue', label='Attendance')
plt.title('Trend in Team Attendance Over Time')
plt.xlabel('Date')
plt.ylabel('Attendance')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()