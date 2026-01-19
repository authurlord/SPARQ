import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'Introduced' and 'Withdrawn' to integers
df['Introduced'] = pd.to_numeric(df['Introduced'])
df['Withdrawn'] = pd.to_numeric(df['Withdrawn'])

# Calculate lifespan
df['Lifespan'] = df['Withdrawn'] - df['Introduced']

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(df['Name'], df['Lifespan'], color='skyblue')
plt.xlabel('Lifespan (Years)')
plt.ylabel('Locomotive Name')
plt.title('Lifespan of Each Locomotive')
plt.gca().invert_yaxis()  # To display the longest lifespan at the top
plt.show()