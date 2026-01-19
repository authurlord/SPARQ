import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['number of examinees'] = pd.to_numeric(df['number of examinees'])
df['pass percentage'] = pd.to_numeric(df['pass percentage'].str.replace('%', ''))

# Calculate correlation coefficient
correlation = df['number of examinees'].corr(df['pass percentage'])

# Plot scatter plot
plt.scatter(df['number of examinees'], df['pass percentage'], color='blue')
plt.title('Pass Percentage vs Number of Examinees')
plt.xlabel('Number of Examinees')
plt.ylabel('Pass Percentage (%)')
plt.grid(True)
plt.show()

print(f"Final Answer: {correlation:.2f}")