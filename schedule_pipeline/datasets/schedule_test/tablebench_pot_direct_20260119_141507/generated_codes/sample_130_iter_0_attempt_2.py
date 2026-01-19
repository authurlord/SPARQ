import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Extract the 'Length' column and convert to numeric
lengths = df['Length'].str.replace(' km', '').astype(float)

# Compute cumulative lengths
cumulative_lengths = lengths.cumsum()

# Create a list of stage names from the 'Stage' column (e.g., SS1, SS2, etc.)
stages = df['Stage']

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(stages, cumulative_lengths, color='skyblue')
plt.xlabel('Stage')
plt.ylabel('Cumulative Length (km)')
plt.title('Cumulative Length at Each Stage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the cumulative lengths for reference
print(cumulative_lengths.tolist())