import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'mpg (US gallons)' to numeric
df['mpg (US gallons)'] = pd.to_numeric(df['mpg (US gallons)'])

# Group by 'Fuel Type' and calculate the mean mpg
avg_mpg = df.groupby('Fuel Type')['mpg (US gallons)'].mean()

# Create a bar chart
plt.figure(figsize=(8, 5))
avg_mpg.plot(kind='bar', color=['blue', 'green'])
plt.title('Average MPG (US gallons) by Fuel Type')
plt.ylabel('Average MPG')
plt.xlabel('Fuel Type')
plt.xticks(rotation=0)
plt.show()