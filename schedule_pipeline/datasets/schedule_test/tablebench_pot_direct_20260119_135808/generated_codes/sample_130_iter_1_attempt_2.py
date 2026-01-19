import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Length' column to float (remove ' km' and convert to numeric)
df['Length'] = df['Length'].str.replace(' km', '').astype(float)

# Calculate cumulative length at each stage
cumulative_length = df['Length'].cumsum()

# Plot bar chart for cumulative lengths
plt.figure(figsize=(12, 6))
plt.bar(df['Stage'], cumulative_length, color='skyblue')
plt.title('Cumulative Length at Each Stage of the Competition')
plt.xlabel('Stage')
plt.ylabel('Cumulative Length (km)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Output the cumulative lengths as final answer
print(f"Final Answer: {', '.join(map(str, cumulative_length))}")