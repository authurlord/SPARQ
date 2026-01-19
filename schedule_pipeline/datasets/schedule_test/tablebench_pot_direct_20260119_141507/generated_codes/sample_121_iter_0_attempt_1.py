import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert Winners and Runner-ups to integers
df['Winners'] = pd.to_numeric(df['Winners'], errors='coerce')
df['Runner-ups'] = pd.to_numeric(df['Runner-ups'], errors='coerce')

# Calculate total number of first and second place finishes
df['Total_Places'] = df['Winners'] + df['Runner-ups']

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df['Club'], df['Total_Places'], color='skyblue')
plt.xlabel('Club')
plt.ylabel('Total First and Second Place Finishes')
plt.title('Number of Times Each Club Has Won First or Second Place')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the total placements for clarity (optional)
print(f"Final Answer: {df['Total_Places'].tolist()}")