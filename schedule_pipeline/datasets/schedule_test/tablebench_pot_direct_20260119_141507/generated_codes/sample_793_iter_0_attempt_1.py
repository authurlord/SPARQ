import pandas as pd

df = pd.read_csv('table.csv')
# Extract the '% of national vote' column and convert to float (remove %)
national_vote_percent = df['% of national vote'].str.replace('%', '').astype(float)
# Calculate standard deviation
std_national_vote = national_vote_percent.std()
print(f"Final Answer: {std_national_vote:.2f}")