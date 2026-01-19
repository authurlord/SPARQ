import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum quantity
max_quantity_row = df.loc[df['quantity'].idxmax()]
make_model = max_quantity_row['make and model']
print(f"Final Answer: {make_model}")