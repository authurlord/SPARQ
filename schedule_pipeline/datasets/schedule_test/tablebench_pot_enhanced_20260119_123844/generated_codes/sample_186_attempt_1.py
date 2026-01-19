import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert '% of popular vote' to float for plotting
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)

# Plot the relationship
plt.figure(figsize=(10, 6))
plt.plot(df['of candidates nominated'], df['% of popular vote'], marker='o', linestyle='-', color='b')
plt.title('Impact of Number of Candidates Nominated on Popular Vote Percentage')
plt.xlabel('Number of Candidates Nominated')
plt.ylabel('% of Popular Vote')
plt.grid(True)
plt.show()