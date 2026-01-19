import pandas as pd

df = pd.read_csv('table.csv')
# Extract the columns of interest
wife_age = df['her age']
children = df['of children']

# Calculate the Pearson correlation coefficient
correlation = wife_age.corr(children)

print(f"Final Answer: {correlation:.2f}")