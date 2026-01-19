import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'mpg (US gallons)' to numeric
df['mpg (US gallons)'] = pd.to_numeric(df['mpg (US gallons)'], errors='coerce')

# Group by fuel type and calculate average mpg
avg_mpg_by_fuel = df.groupby('Fuel Type')['mpg (US gallons)'].mean()

# Create bar chart
plt.figure(figsize=(8, 5))
avg_mpg_by_fuel.plot(kind='bar', color=['blue', 'green'])
plt.title('Average MPG (US gallons) by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Average MPG (US gallons)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Print final answer
print(f"Final Answer: {avg_mpg_by_fuel['diesel']:.1f}, {avg_mpg_by_fuel['petrol']:.1f}")