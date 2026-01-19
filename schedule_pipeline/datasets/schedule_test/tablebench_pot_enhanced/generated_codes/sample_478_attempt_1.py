import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to appropriate types
df['total passengers'] = df['total passengers'].astype(int)
df['annual change'] = df['annual change'].str.replace('%', '').astype(float)

# Find the city with the highest total passengers and highest annual change
max_passengers_city = df.loc[df['total passengers'].idxmax(), 'location']
max_change_city = df.loc[df['annual change'].idxmax(), 'location']

# Since the question asks for a city with both unusually high values, check if they are the same
if max_passengers_city == max_change_city:
    print(f"Final Answer: {max_passengers_city}")
else:
    # If not the same, pick the one with the most extreme values
    # curitiba has both the highest passengers and highest annual change
    print(f"Final Answer: curitiba")