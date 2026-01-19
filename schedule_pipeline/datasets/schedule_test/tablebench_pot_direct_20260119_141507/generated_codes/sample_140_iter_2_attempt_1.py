import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'number of bearers 2009' to float
df['number of bearers 2009'] = df['number of bearers 2009'].str.replace(',', '').astype(float)

# Sort by number of bearers in 2009 in descending order and take top 10
top_10 = df.nlargest(10, 'number of bearers 2009')[['surname', 'number of bearers 2009']]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_10['surname'], top_10['number of bearers 2009'], color='skyblue')
plt.xlabel('Surname')
plt.ylabel('Number of Bearers (2009)')
plt.title('Top 10 Surnames by Number of Bearers in 2009')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the top 10 surnames and their bearer counts for reference
print(top_10.to_string(index=False))