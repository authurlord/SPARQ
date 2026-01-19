# Given data: federal excise tax = 10, total excise tax = 24.5
federal_tax = 10
total_tax = 24.5

# Increase federal excise tax by 20%
new_federal_tax = federal_tax * 1.20

# The non-federal part remains unchanged
non_federal_tax = total_tax - federal_tax

# New total excise tax
new_total_tax = new_federal_tax + non_federal_tax

print(f"Final Answer: {new_total_tax:.2f}")