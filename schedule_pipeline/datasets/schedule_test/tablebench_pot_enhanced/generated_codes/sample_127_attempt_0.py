import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Clean the 'Passengers' column by removing commas and converting to integers
df['Passengers'] = df['Passengers'].str.replace(',', '').astype(int)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df['City'], df['Passengers'], color='skyblue')
plt.title('Number of Passengers for Each City')
plt.xlabel('City')
plt.ylabel('Passengers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()