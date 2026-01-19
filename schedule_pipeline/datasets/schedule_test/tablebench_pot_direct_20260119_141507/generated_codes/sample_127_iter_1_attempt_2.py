import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Passengers' to numeric by removing commas and converting to int
df['Passengers'] = df['Passengers'].str.replace(',', '').astype(int)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(df['City'], df['Passengers'], color='skyblue')
plt.xlabel('City')
plt.ylabel('Number of Passengers')
plt.title('Number of Passengers for Each City')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the passenger count for each city as a final check (optional)
print(f"Final Answer: {df[['City', 'Passengers']].to_string(index=False)}")