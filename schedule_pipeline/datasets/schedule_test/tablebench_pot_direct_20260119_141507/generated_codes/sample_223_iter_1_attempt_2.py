import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'mccain %' column to numeric by removing the '%' symbol
df['mccain %'] = df['mccain %'].str.rstrip('%').astype(float)

# Create a scatter plot to visualize the relationship
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['total'], df['mccain %'], alpha=0.7)
plt.title('McCain Vote Percentage vs Total Votes by County')
plt.xlabel('Total Votes')
plt.ylabel('Percentage of Votes for McCain')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Compute correlation
correlation = df['mccain %'].corr(df['total'])
print(f"Correlation between mccain % and total votes: {correlation:.3f}")