import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'mpg (US gallons)' to numeric for calculation
df['mpg (US gallons)'] = pd.to_numeric(df['mpg (US gallons)'], errors='coerce')

# Group by 'Fuel Type' and calculate mean mpg
avg_mpg_by_fuel = df.groupby('Fuel Type')['mpg (US gallons)'].mean()

# Create a bar chart
plt.figure(figsize=(8, 5))
avg_mpg_by_fuel.plot(kind='bar', color=['blue', 'green'])
plt.title('Average MPG (US gallons) by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Average MPG (US gallons)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()