import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'number of examinees' and 'pass percentage' to numeric
df['number of examinees'] = pd.to_numeric(df['number of examinees'])
df['pass percentage'] = pd.to_numeric(df['pass percentage'].str.rstrip('%'))

# Plot scatter plot to show relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['number of examinees'], df['pass percentage'], color='blue')
plt.title('Relationship Between Number of Examinees and Pass Percentage')
plt.xlabel('Number of Examinees')
plt.ylabel('Pass Percentage (%)')
plt.grid(True)
plt.show()

# Analyze trend: check if higher examinees correlate with higher/lower pass percentage
# Calculate correlation coefficient
correlation = df['number of examinees'].corr(df['pass percentage'])

if correlation > 0:
    trend = "positive"
elif correlation < 0:
    trend = "negative"
else:
    trend = "no clear"

print(f"Final Answer: {trend}")