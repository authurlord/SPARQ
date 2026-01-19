import pandas as pd

df = pd.read_csv('table.csv')
# Find the make and model with the highest quantity
max_quantity_row = df.loc[df['quantity'].idxmax()]
highest_quantity_make_model = max_quantity_row['make and model']
print(f"Final Answer: {highest_quantity_make_model}")