import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Plot spectators over dates
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Spectators'], marker='o', linestyle='-', color='b')
plt.title('Spectators Over Match Dates')
plt.xlabel('Date')
plt.ylabel('Number of Spectators')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()