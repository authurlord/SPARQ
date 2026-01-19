import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert '% of popular vote' to float for plotting
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)

# Plot the relationship
plt.figure(figsize=(10, 6))
plt.plot(df['election'], df['% of popular vote'], marker='o', label='% of Popular Vote')
plt.xlabel('Election Year')
plt.ylabel('% of Popular Vote')
plt.title('Impact of Number of Candidates Nominated on Popular Vote Percentage')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()