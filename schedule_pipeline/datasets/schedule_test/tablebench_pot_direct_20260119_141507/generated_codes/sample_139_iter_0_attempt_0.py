import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert Introduced and Withdrawn to integers
df['Introduced'] = df['Introduced'].astype(str).str.extract(r'(\d+)').astype(int)
df['Withdrawn'] = df['Withdrawn'].astype(str).str.extract(r'(\d+)').astype(int)

# Calculate lifespan
df['lifespan'] = df['Withdrawn'] - df['Introduced']

# Sort by lifespan in descending order for better visualization
df_sorted = df.sort_values(by='lifespan', ascending=False)

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(df_sorted['Name'], df_sorted['lifespan'], color='skyblue')
plt.xlabel('Lifespan (years)')
plt.ylabel('Locomotive Name')
plt.title('Lifespan of Each Locomotive')
plt.gca().invert_yaxis()  # Invert so the longest lifespan is at the top
plt.tight_layout()
plt.show()

# Print the lifespan values for reference
print(f"Final Answer: {df_sorted[['Name', 'lifespan']].to_dict('records')}")