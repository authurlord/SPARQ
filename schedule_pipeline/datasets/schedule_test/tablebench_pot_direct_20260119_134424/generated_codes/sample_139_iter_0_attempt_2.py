import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Calculate lifespan
df['lifespan'] = df['Withdrawn'].astype(int) - df['Introduced'].astype(int)
# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(df['Name'], df['lifespan'], color='skyblue')
plt.xlabel('Lifespan (years)')
plt.ylabel('Locomotive Name')
plt.title('Lifespan of Each Locomotive')
plt.gca().invert_yaxis()  # To display the longest lifespan at the top
plt.show()