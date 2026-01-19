import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['of candidates nominated'] = pd.to_numeric(df['of candidates nominated'])
df['% of popular vote'] = pd.to_numeric(df['% of popular vote'])

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['of candidates nominated'], df['% of popular vote'], color='blue')
plt.title('Impact of Number of Candidates Nominated on Popular Vote Percentage')
plt.xlabel('Number of Candidates Nominated')
plt.ylabel('% of Popular Vote')
plt.grid(True)
plt.show()