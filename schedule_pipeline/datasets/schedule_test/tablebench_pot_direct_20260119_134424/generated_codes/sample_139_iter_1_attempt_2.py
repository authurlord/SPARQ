import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Calculate lifespan for each locomotive
df['lifespan'] = df['Withdrawn'].astype(int) - df['Introduced'].astype(int)
# Sort by lifespan in descending order for better visualization
df_sorted = df.sort_values(by='lifespan', ascending=False)
# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(df_sorted['Name'], df_sorted['lifespan'], color='skyblue')
plt.xlabel('Lifespan (years)')
plt.ylabel('Locomotive Name')
plt.title('Lifespan of Locomotives')
plt.gca().invert_yaxis()  # Invert y-axis to show longest lifespan at the top
plt.tight_layout()
plt.show()