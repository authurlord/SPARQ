import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert pass percentage to float
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Calculate correlation between number of examinees and pass percentage
correlation = df['number of examinees'].astype(int).corr(df['pass percentage'])

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['number of examinees'], df['pass percentage'], color='blue')
plt.title('Pass Percentage vs Number of Examinees (2005–2010)')
plt.xlabel('Number of Examinees')
plt.ylabel('Pass Percentage (%)')
plt.grid(True)
plt.show()

print(f"Final Answer: {correlation:.2f}")