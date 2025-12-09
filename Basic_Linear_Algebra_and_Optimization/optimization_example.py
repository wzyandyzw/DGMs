import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds


def entropy_maximization_analytical(lambda_vec: np.ndarray) -> np.ndarray:
    """
    计算熵最大化问题的解析解
    
    问题描述：
    max_{π} Σ_{i=1}^n λ_i log(π_i)
    s.t. π_i > 0, i = 1, ..., n
         π_i < 1, i = 1, ..., n
         Σ_{i=1}^n π_i = 1
    
    参数:
    lambda_vec: np.ndarray - 输入向量λ，各分量为正
    
    返回:
    np.ndarray - 最优解π*，满足π_i = λ_i / Σλ_j
    """
    # 计算λ向量的和
    sum_lambda = np.sum(lambda_vec)
    
    # 解析解：π_i = λ_i / Σλ_j
    pi_opt = lambda_vec / sum_lambda
    
    return pi_opt


def entropy_maximization_numerical(lambda_vec: np.ndarray) -> np.ndarray:
    """
    使用数值方法求解熵最大化问题
    
    参数:
    lambda_vec: np.ndarray - 输入向量λ，各分量为正
    
    返回:
    np.ndarray - 最优解π*
    """
    n = len(lambda_vec)
    
    # 目标函数：将最大化问题转换为最小化问题，即 -Σλ_i log(π_i)
    def objective(pi):
        return -np.sum(lambda_vec * np.log(pi))
    
    # 约束条件：Σπ_i = 1
    def constraint(pi):
        return np.sum(pi) - 1
    
    # 设置约束
    constraints = [{'type': 'eq', 'fun': constraint}]
    
    # 设置边界条件：0 < π_i < 1
    bounds = Bounds(lb=1e-10, ub=1-1e-10, keep_feasible=True)
    
    # 初始猜测：均匀分布
    pi0 = np.ones(n) / n
    
    # 求解优化问题
    result = minimize(objective, pi0, constraints=constraints, bounds=bounds, method='SLSQP')
    
    if not result.success:
        raise ValueError(f"优化失败：{result.message}")
    
    return result.x


def verify_solution(pi_opt: np.ndarray, lambda_vec: np.ndarray) -> bool:
    """
    验证解是否满足问题的约束条件
    
    参数:
    pi_opt: np.ndarray - 候选解π
    lambda_vec: np.ndarray - 输入向量λ
    
    返回:
    bool - 如果解满足所有约束条件，返回True；否则返回False
    """
    n = len(pi_opt)
    
    # 检查π_i > 0
    if not np.all(pi_opt > 0):
        return False
    
    # 检查π_i < 1
    if not np.all(pi_opt < 1):
        return False
    
    # 检查Σπ_i = 1
    if not np.isclose(np.sum(pi_opt), 1):
        return False
    
    return True


def demonstrate_lagrange_multipliers(pi_opt: np.ndarray, lambda_vec: np.ndarray):
    """
    演示拉格朗日乘数法的结果
    
    在熵最大化问题中，拉格朗日乘数法给出的条件是λ_i / π_i = μ，其中μ是拉格朗日乘数
    """
    print("\n拉格朗日乘数法验证")
    print("==================")
    
    # 计算λ_i / π_i的比值
    ratios = lambda_vec / pi_opt
    
    print(f"λ向量: {lambda_vec}")
    print(f"π*向量: {pi_opt}")
    print(f"λ_i / π_i 的比值: {ratios}")
    print(f"比值的平均值: {np.mean(ratios):.6f}")
    print(f"比值的标准差: {np.std(ratios):.6f}")
    print(f"是否所有比值近似相等: {np.allclose(ratios, np.mean(ratios))}")


if __name__ == "__main__":
    print("知识点7：示例1：优化问题求解")
    print("==============================")
    print("问题描述：")
    print("max_{π} Σ_{i=1}^n λ_i log(π_i)")
    print("s.t. π_i > 0, i = 1, ..., n")
    print("     π_i < 1, i = 1, ..., n")
    print("     Σ_{i=1}^n π_i = 1")
    print()
    print("这是一个熵最大化问题，解析解为：π_i = λ_i / Σλ_j")
    print()
    
    # 创建示例数据
    lambda_vec = np.array([2.0, 3.0, 5.0, 1.0, 4.0])  # 各分量为正
    n = len(lambda_vec)
    
    print(f"输入参数λ = {lambda_vec}")
    print(f"λ向量的长度n = {n}")
    print()
    
    # 使用解析方法求解
    print("1. 解析方法求解：")
    pi_opt_analytical = entropy_maximization_analytical(lambda_vec)
    print(f"最优解π* = {pi_opt_analytical}")
    
    # 验证解
    is_valid = verify_solution(pi_opt_analytical, lambda_vec)
    print(f"解是否有效：{is_valid}")
    print()
    
    # 使用数值方法求解
    print("2. 数值方法求解（使用SciPy的minimize函数）：")
    pi_opt_numerical = entropy_maximization_numerical(lambda_vec)
    print(f"最优解π* = {pi_opt_numerical}")
    
    # 验证解
    is_valid = verify_solution(pi_opt_numerical, lambda_vec)
    print(f"解是否有效：{is_valid}")
    print()
    
    # 比较两种方法的结果
    print("3. 比较解析解和数值解：")
    print(f"两种方法的结果是否一致：{np.allclose(pi_opt_analytical, pi_opt_numerical)}")
    print(f"最大绝对误差：{np.max(np.abs(pi_opt_analytical - pi_opt_numerical)):.10f}")
    print()
    
    # 演示拉格朗日乘数法的结果
    demonstrate_lagrange_multipliers(pi_opt_analytical, lambda_vec)
    print()
    
    # 计算目标函数的最优值
    def objective_value(pi, lambda_vec):
        return np.sum(lambda_vec * np.log(pi))
    
    obj_value = objective_value(pi_opt_analytical, lambda_vec)
    print(f"目标函数的最优值：{obj_value:.6f}")
    
    # 验证约束条件
    print("\n验证约束条件：")
    print(f"1. 所有π_i > 0：{np.all(pi_opt_analytical > 0)}")
    print(f"2. 所有π_i < 1：{np.all(pi_opt_analytical < 1)}")
    print(f"3. Σπ_i = 1：{np.isclose(np.sum(pi_opt_analytical), 1)}")
