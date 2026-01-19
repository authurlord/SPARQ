import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'number of examinees' and 'pass percentage' to numeric
df['number of examinees'] = pd.to_numeric(df['number of examinees'])
df['pass percentage'] = pd.to_numeric(df['pass percentage'].str.rstrip('%'))

# Plot scatter plot to analyze the relationship
plt.scatter(df['number of examinees'], df['pass percentage'], color='blue')
plt.title('Pass Percentage vs Number of Examinees Over the Years')
plt.xlabel('Number of Examinees')
plt.ylabel('Pass Percentage (%)')
plt.grid(True)
plt.show()

# Final answer based on observation: No clear correlation between number of examinees and pass percentage
print("Final Answer: No clear correlation")