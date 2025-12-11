import numpy as np
from scipy.optimize import minimize


def objective_function(x):
    """
    示例目标函数：f(x) = x₁² + x₂²
    """
    return x[0]**2 + x[1]**2


def constraint1(x):
    """
    不等式约束：g₁(x) = x₁ + x₂ - 1 ≤ 0
    """
    return x[0] + x[1] - 1


def constraint2(x):
    """
    等式约束：h₁(x) = x₁ - 0.5 = 0
    """
    return x[0] - 0.5


def solve_with_scipy():
    """
    使用SciPy求解带约束的优化问题
    """
    print("使用SciPy求解带约束的优化问题")
    print("===============================")
    
    # 定义约束条件
    constraints = [
        {'type': 'ineq', 'fun': constraint1},  # 不等式约束 g₁(x) ≤ 0
        {'type': 'eq', 'fun': constraint2}     # 等式约束 h₁(x) = 0
    ]
    
    # 初始猜测
    x0 = [0, 0]
    
    # 求解优化问题，使用SLSQP求解器以获取拉格朗日乘子
    result = minimize(objective_function, x0, constraints=constraints, method='SLSQP')
    
    print(f"优化结果：")
    print(f"最优解 x* = {result.x}")
    print(f"最优值 f(x*) = {result.fun}")
    print(f"成功收敛: {result.success}")
    print(f"迭代次数: {result.nit}")
    print()
    
    return result


def verify_kkt_conditions(result):
    """
    验证KKT条件是否满足
    
    参数:
    result: OptimizeResult - SciPy优化的结果对象
    """
    print("验证KKT条件")
    print("============")
    
    # 获取最优解
    x_star = result.x
    
    # 计算目标函数的梯度
    grad_f = np.array([2 * x_star[0], 2 * x_star[1]])
    
    # 计算约束条件的梯度
    grad_g1 = np.array([1, 1])  # ∇g₁(x) = [1, 1]
    grad_h1 = np.array([1, 0])  # ∇h₁(x) = [1, 0]
    
    # 拉格朗日乘子 - SLSQP求解器将乘子存储在jac属性中
    # jac属性包含目标函数的梯度和约束条件的梯度
    # 对于SLSQP，result.jac的结构是：[目标函数梯度, 不等式约束1梯度, ..., 等式约束1梯度, ...]
    # 但是我们需要的是乘子，它们存储在result.lambda_中
    u1 = result.lambda_[0]  # 不等式约束的乘子
    v1 = result.lambda_[1]  # 等式约束的乘子
    
    print(f"1. 驻点条件：∇f(x*) + u₁∇g₁(x*) + v₁∇h₁(x*) = 0")
    print(f"   ∇f(x*) = {grad_f}")
    print(f"   u₁∇g₁(x*) = {u1 * grad_g1}")
    print(f"   v₁∇h₁(x*) = {v1 * grad_h1}")
    
    sum_terms = grad_f + u1 * grad_g1 + v1 * grad_h1
    print(f"   总和：{sum_terms}")
    print(f"   是否满足驻点条件：{np.allclose(sum_terms, np.zeros_like(sum_terms))}")
    print()
    
    print(f"2. 互补松弛性：u₁g₁(x*) = 0")
    g1_value = constraint1(x_star)
    print(f"   u₁ = {u1}")
    print(f"   g₁(x*) = {g1_value}")
    print(f"   u₁g₁(x*) = {u1 * g1_value}")
    print(f"   是否满足互补松弛性：{np.isclose(u1 * g1_value, 0)}")
    print()
    
    print(f"3. 原始可行性：")
    print(f"   g₁(x*) ≤ 0：{g1_value <= 0 + 1e-8}")
    print(f"   h₁(x*) = 0：{np.isclose(constraint2(x_star), 0)}")
    print()
    
    print(f"4. 对偶可行性：")
    print(f"   u₁ ≥ 0：{u1 >= 0}")


def another_example():
    """
    另一个优化问题示例：f(x) = (x₁ - 1)^2 + (x₂ - 2.5)^2
    约束条件：x₁ - 2x₂ + 2 ≥ 0, x₁² - 2x₂ + 1 ≤ 0, x₁ ≥ 0, x₂ ≥ 0
    """
    print("\n另一个优化问题示例")
    print("==================")
    
    # 目标函数
    def f(x):
        return (x[0] - 1)**2 + (x[1] - 2.5)**2
    
    # 约束条件
    cons = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 2*x[1] + 2},  # x₁ - 2x₂ + 2 ≥ 0
        {'type': 'ineq', 'fun': lambda x: -x[0]**2 + 2*x[1] - 1},  # x₁² - 2x₂ + 1 ≤ 0
        {'type': 'ineq', 'fun': lambda x: x[0]},  # x₁ ≥ 0
        {'type': 'ineq', 'fun': lambda x: x[1]}   # x₂ ≥ 0
    ]
    
    # 初始猜测
    x0 = [0, 0]
    
    # 求解优化问题
    result = minimize(f, x0, constraints=cons)
    
    print(f"最优解 x* = {result.x}")
    print(f"最优值 f(x*) = {result.fun}")
    
    return result


if __name__ == "__main__":
    print("知识点6：Karush-Kuhn-Tucker定理")
    print("=============================")
    print("KKT定理是处理带有不等式约束和等式约束的优化问题的重要工具")
    print()
    print("考虑优化问题：")
    print("min_x f(x) s.t. g_i(x) ≤ 0 和 h_j(x) = 0 对所有 i, j")
    print()
    print("如果x*是局部最小值点，则满足KKT条件：")
    print("1. 驻点条件：∇f(x*) + Σu_i∇g_i(x*) + Σv_j∇h_j(x*) = 0")
    print("2. 互补松弛性：u_i g_i(x*) = 0 对所有 i")
    print("3. 原始可行性：g_i(x*) ≤ 0 和 h_j(x*) = 0 对所有 i, j")
    print("4. 对偶可行性：u_i ≥ 0 对所有 i")
    print()
    
    # 示例1：简单的优化问题
    print("示例1：简单的优化问题")
    print("目标函数：f(x) = x₁² + x₂²")
    print("约束条件：")
    print("  不等式约束：x₁ + x₂ - 1 ≤ 0")
    print("  等式约束：x₁ - 0.5 = 0")
    print()
    
    result = solve_with_scipy()
    verify_kkt_conditions(result)
    
    # 示例2：更复杂的优化问题
    another_example()
