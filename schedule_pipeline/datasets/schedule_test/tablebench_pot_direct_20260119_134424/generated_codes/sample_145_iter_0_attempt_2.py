import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'mpg (US gallons)' to float for numerical operations
df['mpg (US gallons)'] = pd.to_numeric(df['mpg (US gallons)'], errors='coerce')

# Filter data for diesel and petrol
diesel_mpg = df[df['Fuel Type'] == 'diesel']['mpg (US gallons)'].mean()
petrol_mpg = df[df['Fuel Type'] == 'petrol']['mpg (US gallons)'].mean()

# Create a bar chart
fuel_types = ['Diesel', 'Petrol']
avg_mpg = [diesel_mpg, petrol_mpg]

plt.figure(figsize=(8, 5))
plt.bar(fuel_types, avg_mpg, color=['blue', 'red'])
plt.title('Average MPG (US gallons) by Fuel Type')
plt.ylabel('Average MPG')
plt.show()

print(f"Final Answer: {diesel_mpg:.1f}, {petrol_mpg:.1f}")