"""Math agent specialized in solving mathematical problems with self-improvement."""

import numpy as np
import sympy as sp
from typing import Any, Dict
from agents.base_agent import BaseAgent, Strategy


class MathAgent(BaseAgent):
    """Agent for mathematical computations with strategy evolution."""

    def __init__(self, name: str = "MathAgent", memory_path: str = "memory"):
        super().__init__(name, memory_path)

        # Initialize with multiple solving strategies
        self.create_strategy("symbolic", {"method": "sympy", "simplify": True})
        self.create_strategy("numeric", {"method": "numpy", "precision": 1e-10})
        self.create_strategy("hybrid", {"method": "both", "threshold": 100})

    def execute_task(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Execute mathematical task using selected strategy."""
        problem_type = task.get("type", "equation")

        if problem_type == "equation":
            return self._solve_equation(task, strategy)
        elif problem_type == "optimization":
            return self._solve_optimization(task, strategy)
        elif problem_type == "differential":
            return self._solve_differential(task, strategy)
        elif problem_type == "integration":
            return self._solve_integration(task, strategy)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    def _solve_equation(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve algebraic equations."""
        equation = task.get("equation")
        variable = task.get("variable", "x")

        method = strategy.parameters.get("method")

        # Check for force_numeric flag in task (overrides strategy selection)
        force_numeric = task.get("force_numeric", False)
        if force_numeric:
            method = "numpy"

        # Hybrid strategy: choose based on equation complexity
        if method == "both":
            # Parse equation to check complexity
            x = sp.Symbol(variable)
            expr = sp.sympify(equation)

            # Decide which method to use based on equation complexity
            # Use numeric for high-degree polynomials or transcendental equations
            degree = 0
            try:
                if expr.is_polynomial(x):
                    poly = sp.Poly(expr, x)
                    degree = poly.degree()
            except:
                # Not a polynomial, likely transcendental
                degree = 100  # Force numeric for transcendental

            # Use numeric for high-degree (>4) or transcendental functions
            if degree > 4 or any(func in str(expr) for func in ['sin', 'cos', 'tan', 'exp', 'log']):
                method = "numpy"
            else:
                method = "sympy"

        if method == "sympy":
            x = sp.Symbol(variable)
            # Parse equation
            expr = sp.sympify(equation)
            solutions = sp.solve(expr, x)

            if strategy.parameters.get("simplify", False):
                solutions = [sp.simplify(sol) for sol in solutions]

            return {
                "method": "symbolic",
                "solutions": [str(sol) for sol in solutions]
            }

        elif method == "numpy":
            # For numerical solving, we'd use scipy
            from scipy.optimize import fsolve

            # Convert sympy to lambda function
            x = sp.Symbol(variable)
            expr = sp.sympify(equation)
            f = sp.lambdify(x, expr, "numpy")

            # Try multiple starting points
            starting_points = [-10, -1, 0, 1, 10]
            solutions = []

            for x0 in starting_points:
                try:
                    sol = fsolve(f, x0, full_output=True)
                    if sol[2] == 1:  # Solution converged
                        solutions.append(float(sol[0][0]))
                except:
                    continue

            # Remove duplicates
            unique_solutions = []
            tol = strategy.parameters.get("precision", 1e-10)
            for sol in solutions:
                if not any(abs(sol - u) < tol for u in unique_solutions):
                    unique_solutions.append(sol)

            return {
                "method": "numeric",
                "solutions": unique_solutions
            }

    def _solve_optimization(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve optimization problems."""
        from scipy.optimize import minimize

        objective = task.get("objective")
        constraints = task.get("constraints", [])
        bounds = task.get("bounds")

        # Convert to lambda function
        x = sp.Symbol('x')
        y = sp.Symbol('y') if 'y' in objective else None

        obj_expr = sp.sympify(objective)

        if y:
            obj_func = sp.lambdify([x, y], obj_expr, "numpy")
            x0 = [0, 0]
        else:
            obj_func = sp.lambdify(x, obj_expr, "numpy")
            x0 = [0]

        result = minimize(obj_func, x0, method='SLSQP', bounds=bounds)

        return {
            "method": "scipy.optimize",
            "optimal_point": result.x.tolist(),
            "optimal_value": float(result.fun),
            "success": result.success
        }

    def _solve_differential(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve differential equations."""
        equation = task.get("equation")
        function = task.get("function", "y")
        variable = task.get("variable", "x")
        initial_conditions = task.get("initial_conditions", {})

        # Symbolic solution
        x = sp.Symbol(variable)
        y = sp.Function(function)

        # Parse differential equation
        eq = sp.sympify(equation)

        # Solve
        if initial_conditions:
            ics = {sp.sympify(k): v for k, v in initial_conditions.items()}
            solution = sp.dsolve(eq, y(x), ics=ics)
        else:
            solution = sp.dsolve(eq, y(x))

        return {
            "method": "symbolic",
            "solution": str(solution),
            "symbolic_form": solution
        }

    def _solve_integration(self, task: Dict[str, Any], strategy: Strategy) -> Any:
        """Solve integration problems."""
        integrand = task.get("integrand")
        variable = task.get("variable", "x")
        limits = task.get("limits", None)  # (a, b) for definite integral

        x = sp.Symbol(variable)
        expr = sp.sympify(integrand)

        if limits:
            # Definite integral
            a, b = limits
            result = sp.integrate(expr, (x, a, b))
            numeric_val = None
            try:
                numeric_val = float(result.evalf())
            except (ValueError, TypeError, AttributeError):
                # Symbolic result cannot be evaluated to float (e.g., contains symbols)
                numeric_val = None
            return {
                "method": "symbolic",
                "type": "definite",
                "result": str(result),
                "numeric_value": numeric_val
            }
        else:
            # Indefinite integral
            result = sp.integrate(expr, x)
            return {
                "method": "symbolic",
                "type": "indefinite",
                "result": str(result) + " + C"
            }

    def evaluate_result(self, task: Dict[str, Any], output: Any) -> Dict[str, float]:
        """Evaluate the quality of the solution."""
        metrics = {}

        # Check if solution exists
        if output and "solutions" in output:
            metrics["found_solution"] = 1.0

            # Verify solution by substitution
            if task.get("type") == "equation":
                equation = task.get("equation")
                variable = task.get("variable", "x")
                x = sp.Symbol(variable)
                expr = sp.sympify(equation)

                verified = 0
                solutions = output.get("solutions", [])

                for sol_str in solutions:
                    try:
                        sol = sp.sympify(sol_str)
                        value = expr.subs(x, sol)
                        if abs(complex(value)) < 1e-10:
                            verified += 1
                    except (ValueError, TypeError, AttributeError, SyntaxError):
                        # Solution verification failed (symbolic or complex expression)
                        continue

                if solutions:
                    metrics["accuracy"] = verified / len(solutions)
                else:
                    metrics["accuracy"] = 0.0
        else:
            metrics["found_solution"] = 0.0
            metrics["accuracy"] = 0.0

        return metrics

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default strategy parameters."""
        return {
            "method": "sympy",
            "simplify": True,
            "precision": 1e-10
        }
