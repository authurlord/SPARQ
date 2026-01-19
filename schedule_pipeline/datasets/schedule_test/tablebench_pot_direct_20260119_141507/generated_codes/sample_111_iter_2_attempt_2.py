import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'Date' to datetime for proper plotting
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# Plot line chart of spectators over dates
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Spectators'], marker='o', linestyle='-', color='b')
plt.title('Spectators vs Date of Matches')
plt.xlabel('Date')
plt.ylabel('Number of Spectators')
plt.grid(True)
plt.tight_layout()
plt.show()