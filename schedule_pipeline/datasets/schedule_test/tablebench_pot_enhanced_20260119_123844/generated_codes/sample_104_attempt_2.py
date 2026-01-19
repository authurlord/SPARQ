import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Gap' to numeric, removing the '+' sign and handling '–' for first place
df['Gap'] = df['Gap'].str.replace('+', '').str.strip()
df['Gap'] = pd.to_numeric(df['Gap'], errors='coerce')

# Replace '–' with 0 for the first-place finisher
df['Gap'] = df['Gap'].fillna(0)

# Create a line chart
plt.figure(figsize=(12, 6))
plt.plot(df['Pos'], df['Gap'], marker='o', linestyle='-', color='b')
plt.title('Time Difference Between Competitors and First-Place Finisher')
plt.xlabel('Position (Pos)')
plt.ylabel('Gap (seconds)')
plt.grid(True)
plt.xticks(df['Pos'])
plt.tight_layout()
plt.show()

# Final Answer: List of gap values in order
print("Final Answer: " + ", ".join(map(str, df['Gap'].astype(int))))