import pandas as pd

df = pd.read_csv('table.csv')
# Filter states where males rank is higher than 10
filtered_df = df[df['males rank'].astype(int) > 10]
# Find the state with the highest females (%) value
max_females_state = filtered_df.loc[filtered_df['females (%)'].idxmax()]
print(f"Final Answer: {max_females_state['state']}")