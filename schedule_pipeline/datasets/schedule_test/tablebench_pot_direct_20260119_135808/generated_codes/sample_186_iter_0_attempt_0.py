import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'of candidates nominated' and '% of popular vote' to numeric
df['of candidates nominated'] = pd.to_numeric(df['of candidates nominated'])
df['% of popular vote'] = pd.to_numeric(df['% of popular vote'])

# Plot the relationship
plt.figure(figsize=(10, 6))
plt.plot(df['election'], df['% of popular vote'], marker='o', label='% of Popular Vote')
plt.plot(df['election'], df['of candidates nominated'], marker='s', label='Number of Candidates Nominated')
plt.title('Impact of Number of Candidates Nominated on Popular Vote Over Time')
plt.xlabel('Election Year')
plt.ylabel('Value')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Final Answer: The data shows that as the number of candidates nominated increases, the percentage of popular vote generally increases, indicating a positive correlation.
Final Answer: Positive Correlation