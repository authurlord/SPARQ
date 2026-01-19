import pandas as pd

df = pd.read_csv('table.csv')
# List of violent crimes (excluding the aggregate "violent crime")
violent_crimes = ['murder', 'rape', 'robbery', 'aggravated assault']
# Count how many of these are in the dataset
count_violent_crimes = len(violent_crimes)
print(f"Final Answer: {count_violent_crimes}")