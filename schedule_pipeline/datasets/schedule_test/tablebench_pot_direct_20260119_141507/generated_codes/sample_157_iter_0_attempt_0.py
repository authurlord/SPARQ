import pandas as pd

df = pd.read_csv('table.csv')

# Select numerical columns for correlation
numeric_cols = ['revenue (millions)', 'employees', 'profit (millions)']
df_numeric = df[numeric_cols].dropna()

# Calculate correlation between profit and revenue, and profit and employees
corr_revenue_profit = df_numeric['profit (millions)'].corr(df_numeric['revenue (millions)'])
corr_employees_profit = df_numeric['profit (millions)'].corr(df_numeric['employees'])

# Check if either correlation is strong (absolute value > 0.5)
if abs(corr_revenue_profit) > 0.5 or abs(corr_employees_profit) > 0.5:
    if abs(corr_revenue_profit) > 0.5:
        result = "revenue (millions)"
    elif abs(corr_employees_profit) > 0.5:
        result = "employees"
    else:
        result = "no clear impact"
else:
    result = "no clear impact"

print(f"Final Answer: {result}")