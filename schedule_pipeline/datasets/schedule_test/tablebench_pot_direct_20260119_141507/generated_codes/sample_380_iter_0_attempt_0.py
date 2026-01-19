import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'crime' is a violent crime
violent_crimes = ['murder', 'rape', 'robbery', 'aggravated assault', 'violent crime']
count_violent_crimes = df[df['crime'].isin(violent_crimes)]['crime'].count()
print(f"Final Answer: {count_violent_crimes}")