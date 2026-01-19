import pandas as pd

# Load the data from the JSON table
df = pd.read_csv('table.csv')

# Calculate the average number of tropical cyclones per season
average_cyclones = df['tropical cyclones'].mean()

# Output the final answer
print(f"Final Answer: {average_cyclones:.1f}")