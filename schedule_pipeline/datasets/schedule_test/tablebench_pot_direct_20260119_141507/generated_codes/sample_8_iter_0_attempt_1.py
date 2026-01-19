import pandas as pd

df = pd.read_csv('table.csv')
# Extract the '% of national vote' column and convert percentages to float
national_vote_percent = df['% of national vote'].str.replace('%', '').astype(float)
average_national_vote = national_vote_percent.mean()
print(f"Final Answer: {average_national_vote:.2f}")