import pandas as pd

df = pd.read_csv('table.csv')
# Extract the relevant columns
kcal_upper = df['upper index kcal / nm 3']
mj_upper = df['upper index mj / nm 3']

# Calculate the correlation coefficient
correlation = kcal_upper.corr(mj_upper)
print(f"Final Answer: {correlation:.3f}")