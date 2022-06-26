from ortools.linear_solver import pywraplp
import numpy as np
from . import match_service

def solve():
    results = match_service._get_labelled_edges()

    # Create a solver using the GLOP backend
    solver = pywraplp.Solver('Maximize objective', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Variables: weights
    variables = {}
    matchers = list(results[0]['scores'].keys())
    for matcher in matchers:
        temp = solver.NumVar(0, solver.infinity(),  matcher)
        variables[matcher] = temp

    correct_results = list(map(lambda result: result['scores'], filter(lambda result: result['correct'], results)))
    false_results = list(map(lambda result: result['scores'], filter(lambda result: not result['correct'], results)))

    # Constraints
    solver.Add(solver.Sum(variables.values()) == 1)
    for result in false_results:
        # scores = result
        solver.Add(solver.Sum(variable * result[name] for (name, variable) in variables.items()) <= 0.5)
        
        # constraint.SetCoefficient(np.array(list(variables.values()))   , scores)
        # solver.Add(list(variables.values()) @ scores <= 0.6)
        # print("TESTING", list(variables.values()) @ scores)

    # ct = solver.Constraint(1, 'ct')
    # Objective Function
    objective = solver.Objective()
    for matcher in matchers:
        value = sum(list(map(lambda score: score[matcher], correct_results)))

        print("OBJECTIVE: ", variables[matcher], value)
        objective.SetCoefficient(variables[matcher], value)
    objective.SetMaximization()

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        for name, variable in variables.items():
            print(name, ' =', variable.solution_value())
    else:
        print('The problem does not have an optimal solution.')
    print(status)
    print("END OF FUNCTION")

# Old
# def solve():
#     results = match_service._get_labelled_edges()

#     # Create a solver using the GLOP backend
#     solver = pywraplp.Solver('Maximize objective', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

#     # Variables: weights
#     variables = {}
#     matchers = list(results[0]['scores'].keys())
#     for matcher in matchers:
#         temp = solver.NumVar(0, solver.infinity(),  matcher)
#         variables[matcher] = temp

#     # Constraints
#     solver.Add(solver.Sum(variables.values()) == 1)
#     # ct = solver.Constraint(1, 'ct')

#     processed_results = list(map(lambda result: result['scores'] if result['correct'] else {x: 1 - y for (x, y) in result['scores'].items()}, results))
#     # Objective Function
#     objective = solver.Objective()
#     for matcher in matchers:
#         value = sum(list(map(lambda score: score[matcher], processed_results)))

#         print("OBJECTIVE: ", variables[matcher], value)
#         objective.SetCoefficient(variables[matcher], value)
#     objective.SetMaximization()

#     status = solver.Solve()
#     if status == pywraplp.Solver.OPTIMAL:
#         print('Solution:')
#         print('Objective value =', solver.Objective().Value())
#         for name, variable in variables.items():
#             print(name, ' =', variable.solution_value())
#     else:
#         print('The problem does not have an optimal solution.')
#     print(status)
#     print("END OF FUNCTION")

