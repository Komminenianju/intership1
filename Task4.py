# Import PuLP
from pulp import LpMaximize, LpProblem, LpVariable, value

# Define the problem
model = LpProblem("Product-Mix-Optimization", LpMaximize)

# Decision variables
A = LpVariable("Product_A", lowBound=0, cat='Continuous')
B = LpVariable("Product_B", lowBound=0, cat='Continuous')

# Objective function
model += 40 * A + 50 * B, "Total_Profit"

# Constraints
model += 2 * A + 1 * B <= 100, "Machine_1_Hours"
model += 1 * A + 2 * B <= 80, "Machine_2_Hours"

# Solve the problem
model.solve()

# Results
print(f"Produce {A.varValue:.2f} units of Product A")
print(f"Produce {B.varValue:.2f} units of Product B")
print(f"Maximum Profit: ₹{value(model.objective):.2f}")