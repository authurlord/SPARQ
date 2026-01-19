import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Sort by date for proper line chart order
df = df.sort_values('Date')

# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Spectators'], marker='o', linestyle='-', color='b')
plt.title('Spectators vs Match Dates')
plt.xlabel('Date')
plt.ylabel('Number of Spectators')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()