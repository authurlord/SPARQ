import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rating' to numeric and calculate mean and median
ratings = pd.to_numeric(df['rating'], errors='coerce')
mean_rating = ratings.mean()
median_rating = ratings.median()
print(f"Final Answer: {mean_rating:.2f}, {median_rating:.2f}")