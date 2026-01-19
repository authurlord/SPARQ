import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Select relevant columns and sort by 'number of bearers 2009' in descending order
top_10_2009 = df.sort_values(by='number of bearers 2009', ascending=False).head(10)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_10_2009['surname'], top_10_2009['number of bearers 2009'])
plt.xlabel('Surname')
plt.ylabel('Number of Bearers (2009)')
plt.title('Top 10 Surnames by Number of Bearers in 2009')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()