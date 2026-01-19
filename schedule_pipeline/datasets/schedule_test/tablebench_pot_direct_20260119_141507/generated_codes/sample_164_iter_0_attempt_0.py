import pandas as pd

df = pd.read_csv('table.csv')

# Check correlation between Labour councillors and other parties
correlation_with_cons = df['Labour councillors'].corr(df['Conservative councillors'])
correlation_with_ind = df['Labour councillors'].corr(df['Independent councillors'])
correlation_with_lib = df['Labour councillors'].corr(df['Liberal councillors'])

# Check trend over years
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
trend = df.groupby('Year')['Labour councillors'].mean()

# If correlation is weak (close to 0) and no clear trend, then no clear impact
if (abs(correlation_with_cons) < 0.2 and 
    abs(correlation_with_ind) < 0.2 and 
    abs(correlation_with_lib) < 0.2 and 
    trend.std() < 1):
    print("Final Answer: no clear impact")
else:
    # Identify which factors show significant influence
    factors = []
    if abs(correlation_with_cons) > 0.2:
        factors.append("Conservative councillors")
    if abs(correlation_with_ind) > 0.2:
        factors.append("Independent councillors")
    if abs(correlation_with_lib) > 0.2:
        factors.append("Liberal councillors")
    if trend.std() > 1:
        factors.append("Year")
    
    if factors:
        print(f"Final Answer: {', '.join(factors)}")
    else:
        print("Final Answer: no clear impact")