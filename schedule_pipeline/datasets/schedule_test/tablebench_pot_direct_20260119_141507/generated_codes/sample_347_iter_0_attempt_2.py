import pandas as pd

df = pd.read_csv('table.csv')
# Filter episodes with rating >= 5.3 and count them
count_high_rating = df[df['rating'] >= '5.3'].shape[0]
print(f"Final Answer: {count_high_rating}")