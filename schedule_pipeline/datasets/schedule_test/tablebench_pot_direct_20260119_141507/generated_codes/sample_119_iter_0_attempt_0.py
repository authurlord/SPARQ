import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert Date to datetime for proper plotting
df['Date'] = pd.to_datetime(df['Date'])

# Plot attendance over time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Attendance'], marker='o', linestyle='-', color='b')
plt.title('Trend in Team Attendance Over Time')
plt.xlabel('Date')
plt.ylabel('Attendance')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()