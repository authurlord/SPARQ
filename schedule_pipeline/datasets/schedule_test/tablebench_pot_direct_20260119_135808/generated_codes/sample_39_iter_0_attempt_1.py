import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'camp mates' column to integer and sum all values
total_camp_mates = df['camp mates'].astype(int).sum()
print(f"Final Answer: {total_camp_mates}")