import pandas as pd

df = pd.read_csv('table.csv')
# Select only the judge score columns
judge_columns = ['jaanu någisto', 'iiris vesik', 'erik morna', 'veronika portsmuth', 'chalice', 'kristo rajasaare', 'hannaliisa uusmaa', 'siim nestor', 'peeter vã¤hi']
judge_scores = df[judge_columns]

# Calculate standard deviation for each judge
std_devs = judge_scores.std()

# Find the judge with the highest standard deviation
max_std_judge = std_devs.idxmax()

print(f"Final Answer: {max_std_judge}")