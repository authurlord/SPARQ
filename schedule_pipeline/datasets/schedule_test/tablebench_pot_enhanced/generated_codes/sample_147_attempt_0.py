import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert pass percentage to float by removing the '%' sign and converting to float
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Calculate correlation between number of examinees and pass percentage
correlation = df['number of examinees'].astype(int).corr(df['pass percentage'])

# Optional: Plot scatter plot to visualize the relationship
plt.scatter(df['number of examinees'].astype(int), df['pass percentage'])
plt.title('Pass Percentage vs Number of Examinees')
plt.xlabel('Number of Examinees')
plt.ylabel('Pass Percentage (%)')
plt.grid(True)
plt.show()

print(f"Final Answer: {correlation:.2f}")