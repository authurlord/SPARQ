import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'Passengers' to numeric by removing commas
df['Passengers'] = df['Passengers'].str.replace(',', '').astype(int)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df['City'], df['Passengers'], color='skyblue')
plt.xlabel('City')
plt.ylabel('Number of Passengers')
plt.title('Number of Passengers for Each City')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Answer is not a value but a chart — however, the instruction requires final answer in format
# Since no specific value is requested, and the task is to draw the chart, we just confirm the action.
Final Answer: chart