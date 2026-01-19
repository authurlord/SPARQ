import pandas as pd

df = pd.read_csv('table.csv')

# Compute correlation with 'viewers (in millions)'
correlations = df.corr()['viewers (in millions)'].drop('viewers (in millions)')

# Identify factors with absolute correlation > 0.3
significant_factors = [col for col in correlations.index if abs(correlations[col]) > 0.3]

if significant_factors:
    final_answer = ", ".join(significant_factors)
else:
    final_answer = "no clear impact"

print(f"Final Answer: {final_answer}")