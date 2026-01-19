import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert Introduced and Withdrawn to integers
df['Introduced'] = pd.to_numeric(df['Introduced'], errors='coerce')
df['Withdrawn'] = pd.to_numeric(df['Withdrawn'], errors='coerce')

# Calculate lifespan
df['lifespan'] = df['Withdrawn'] - df['Introduced']

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(df['Name'], df['lifespan'], color='skyblue')
plt.xlabel('Lifespan (years)')
plt.ylabel('Locomotive Name')
plt.title('Lifespan of Each Locomotive')
plt.gca().invert_yaxis()  # Optional: to have newer ones at top
plt.tight_layout()
plt.show()

# Print the lifespan values for reference (optional)
print(df[['Name', 'lifespan']])