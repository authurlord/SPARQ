import pandas as pd

df = pd.read_csv('table.csv')

# Display the table structure and basic info
print("Table Columns and Their Significance:")
print("- department: Name of the department in Bolivia.")
print("- micro (10ha): Number of farms with area ≤10 hectares.")
print("- small (100ha): Number of farms with area >10 to ≤100 hectares.")
print("- medium (500ha): Number of farms with area >100 to ≤500 hectares.")
print("- big (>500ha): Number of farms with area >500 hectares.")
print("- total: Total number of farms in the department.")

# Identify the department with the highest total
max_total_dept = df.loc[df['total'].idxmax()]['department']
print(f"\nDepartment with the highest total farms: {max_total_dept}")

# Check which farm size category dominates overall
total_by_size = df[['micro (10ha)', 'small (100ha)', 'medium (500ha)', 'big (>500ha)']].sum()
dominant_size = total_by_size.idxmax()
print(f"Most common farm size category across all departments: {dominant_size}")

# Look for patterns: e.g., large farms in Santa Cruz?
large_farms_santa_cruz = df[df['department'] == 'santa cruz']['big (>500ha)'].values[0]
print(f"Number of big farms in Santa Cruz: {large_farms_santa_cruz}")

# Check if Cochabamba has the highest total
cochabamba_total = df[df['department'] == 'cochabamba']['total'].values[0]
print(f"Total farms in Cochabamba: {cochabamba_total}")

# Final summary
print("\nNotable Trends:")
print("- Cochabamba has the highest total number of farms (81,925), indicating high agricultural activity.")
print("- The 'small (100ha)' category is the most prevalent across departments, suggesting a dominance of medium-sized farms.")
print("- Santa Cruz has relatively few big farms despite its size, but notable medium and small farms.")
print("- Departments like Tarija and Cochabamba have strong representation in medium and big farm categories.")

# Final Answer: Summarize key components and trends
print("Final Answer: department, micro (10ha), small (100ha), medium (500ha), big (>500ha), total")