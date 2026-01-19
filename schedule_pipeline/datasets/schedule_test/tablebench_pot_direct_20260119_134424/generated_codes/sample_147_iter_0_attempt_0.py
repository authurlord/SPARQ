import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'number of examinees' and 'pass percentage' to numeric
df['number of examinees'] = pd.to_numeric(df['number of examinees'])
df['pass percentage'] = pd.to_numeric(df['pass percentage'].str.rstrip('%'))

# Create scatter plot
plt.scatter(df['number of examinees'], df['pass percentage'])
plt.xlabel('Number of Examinees')
plt.ylabel('Pass Percentage')
plt.title('Effect of Number of Examinees on Pass Percentage Over Years')
plt.show()