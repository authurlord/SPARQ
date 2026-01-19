import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Length' column to numeric (removing any non-numeric characters if necessary)
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')

# Calculate cumulative sum of lengths
cumulative_lengths = df['Length'].cumsum()

# Plot bar chart for cumulative lengths at each stage
plt.figure(figsize=(12, 6))
plt.bar(df['Stage'], cumulative_lengths, color='skyblue')
plt.title('Cumulative Lengths at Each Stage of the Competition')
plt.xlabel('Stage')
plt.ylabel('Cumulative Length (km)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Output the final answer as requested
print(f"Final Answer: {', '.join(map(str, cumulative_lengths))}")