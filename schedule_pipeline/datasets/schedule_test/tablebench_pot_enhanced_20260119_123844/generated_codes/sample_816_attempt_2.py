import pandas as pd

df = pd.read_csv('table.csv')
# Select only the score columns for judges
judge_columns = ['jaanu någisto', 'iiris vesik', 'erik morna', 'veronika portsmuth', 'chalice', 'kristo rajasaare', 'hannaliisa uusmaa', 'siim nestor', 'peeter vã¤hi', 'helen sildna', 'ott lepland']
# Calculate standard deviation for each judge
std_devs = df[judge_columns].astype(int).std()
# Find the judge with the highest standard deviation
max_std_judge = std_devs.idxmax()
print(f"Final Answer: {max_std_judge}")