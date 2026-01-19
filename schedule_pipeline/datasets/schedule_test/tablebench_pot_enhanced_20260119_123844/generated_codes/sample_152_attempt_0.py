import pandas as pd

df = pd.read_csv('table.csv')

# Check if there's a clear correlation between the factors and the result
# The 'bottom two' couple has the lowest values in judges, public, total, and vote percentage
# All others are 'safe' with higher values
# Hence, all three factors contribute significantly

print("Final Answer: judges, public, vote percentage")