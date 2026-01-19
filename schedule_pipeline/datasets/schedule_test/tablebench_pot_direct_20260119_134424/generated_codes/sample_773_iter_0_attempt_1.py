import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Plot the trend
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['total revenue'], marker='o', linestyle='-', color='b')
plt.title('Total Revenue Trend Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Perform linear regression to project next year's revenue
X = np.array(df['year'].astype(int)).reshape(-1, 1)
y = df['total revenue'].values

model = np.polyfit(X.flatten(), y, 1)
predicted_next_year = np.polyval(model, X[-1] + 1)

print(f"Final Answer: Increasing trend, {predicted_next_year:.0f}")