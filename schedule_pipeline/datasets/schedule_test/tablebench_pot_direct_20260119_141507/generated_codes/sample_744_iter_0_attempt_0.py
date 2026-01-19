import pandas as pd

df = pd.read_csv('table.csv')

# Extract years and values
years = [1983, 1986, 1991, 1996, 2001, 2005, 2009]
candidates = [4, 9, 42, 71, 72, 79, 85]
popular_vote = [0.19, 0.24, 0.86, 1.99, 12.39, 9.17, 8.21]

# Project candidates fielded: linear trend from 1983 to 2009
# Total years: 26, increase: 85 - 4 = 81
# Annual increase: 81 / 26 ≈ 3.115
# Next election after 2009: assume 2015 (6 years later)
next_candidates = 85 + (6 * (81 / 26))

# Popular vote: peaks at 2001, then declines. The peak is 12.39%, so we assume it remains near that value.
next_popular_vote = 12.39

print(f"Final Answer: {int(next_candidates)}, {next_popular_vote:.2f}")