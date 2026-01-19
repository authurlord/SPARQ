import pandas as pd

df = pd.read_csv('table.csv')

total_years = 0
current_year = 2024

for index, row in df.iterrows():
    taking_office = int(row['Taking Office'])
    leaving = row['Leaving']
    
    if leaving == 'Current':
        # Serve until current year
        years_served = current_year - taking_office
    else:
        leaving_year = int(leaving)
        years_served = leaving_year - taking_office
    
    total_years += years_served

print(f"Final Answer: {total_years}")