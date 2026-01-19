import pandas as pd

df = pd.read_csv('table.csv')
# Extract the '% of national vote' column and convert to float (remove %)
national_vote_percent = df['% of national vote'].str.replace('%', '').astype(float)
# Calculate the mean
mean_national_vote = national_vote_percent.mean()
print(f"Final Answer: {mean_national_vote:.2f}")