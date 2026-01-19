import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Sort by 'number of bearers 2009' in descending order and take top 10
top_10 = df.sort_values(by='number of bearers 2009', ascending=False).head(10)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_10['surname'], top_10['number of bearers 2009'], color='skyblue')
plt.xlabel('Surname')
plt.ylabel('Number of Bearers (2009)')
plt.title('Top 10 Surnames by Number of Bearers in 2009')
plt.xticks(rotation=45)

# Label each bar with the surname
for i, v in enumerate(top_10['number of bearers 2009']):
    plt.text(i, v + 1000, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()