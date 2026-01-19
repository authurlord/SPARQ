import pandas as pd

df = pd.read_csv('table.csv')
# List of violent crimes
violent_crimes = ['murder', 'rape', 'robbery', 'aggravated assault', 'violent crime']
# Count the number of violent crime types
count_violent_crimes = len(violent_crimes)
print(f"Final Answer: {count_violent_crimes}")