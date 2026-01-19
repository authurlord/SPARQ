import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert pass percentage from string with '%' to float
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Create a scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df['number of examinees'], df['pass percentage'], color='blue')
plt.title('Number of Examinees vs Pass Percentage Over the Years')
plt.xlabel('Number of Examinees')
plt.ylabel('Pass Percentage (%)')
plt.grid(True)
plt.show()

# Final answer: Based on the visualization, there is no clear direct effect; however, we can state the trend in simple terms.
# Since the question asks "how does it affect", we describe the observed pattern.
Final Answer: No clear direct effect