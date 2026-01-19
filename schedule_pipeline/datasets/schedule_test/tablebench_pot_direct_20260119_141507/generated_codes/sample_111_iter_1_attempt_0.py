import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Date' to datetime for proper plotting
df['Date'] = pd.to_datetime(df['Date'])

# Extract the 'Spectators' column and convert to integer
df['Spectators'] = df['Spectators'].str.replace(',', '').astype(int)

# Create a line chart of spectators over dates
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Spectators'], marker='o', linestyle='-', color='b')
plt.title('Spectators Over Match Dates')
plt.xlabel('Date')
plt.ylabel('Number of Spectators')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()