import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the 'Gap' column (time difference from first place)
gaps = df['Gap'].dropna()  # Remove any NaN values
drivers = df['Driver'].dropna()

# Create a line chart of gaps vs driver names
plt.figure(figsize=(12, 6))
plt.plot(drivers, gaps, marker='o', linestyle='-', color='b')
plt.title('Time Difference Between Competitors and First-Place Finisher')
plt.xlabel('Driver')
plt.ylabel('Time Difference (seconds)')
plt.xticks(rotation=90)  # Rotate labels to avoid overlap
plt.grid(True)
plt.tight_layout()
plt.show()