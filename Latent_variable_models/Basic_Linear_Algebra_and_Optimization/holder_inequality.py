import numpy as np


def p_norm(vector: np.ndarray, p: float) -> float:
    """
    计算向量的p-范数
    
    参数:
    vector: np.ndarray - 输入向量
    p: float - 范数的阶数，1 ≤ p ≤ ∞
    
    返回:
    float - 向量的p-范数
    """
    if p == np.inf:
        # 无穷范数：向量元素的最大绝对值
        return np.max(np.abs(vector))
    elif p == 1:
        # 1-范数：向量元素绝对值之和
        return np.sum(np.abs(vector))
    else:
        # p-范数：(Σ|x_i|^p)^(1/p)
        return np.power(np.sum(np.power(np.abs(vector), p)), 1/p)


def verify_holder_inequality(x: np.ndarray, y: np.ndarray, p: float, q: float, tol: float = 1e-8) -> bool:
    """
    验证霍尔德不等式是否成立
    
    不等式形式：Σ|x_i y_i| ≤ ||x||_p · ||y||_q
    其中1/p + 1/q = 1
    
    参数:
    x: np.ndarray - 第一个向量
    y: np.ndarray - 第二个向量
    p: float - 第一个范数的阶数
    q: float - 第二个范数的阶数
    tol: float - 容差，用于比较不等式两边
    
    返回:
    bool - 如果不等式成立，返回True；否则返回False
    """
    # 检查1/p + 1/q = 1
    if not np.isclose(1/p + 1/q, 1):
        raise ValueError(f"参数p和q不满足1/p + 1/q = 1的条件")
    
    # 计算不等式左边：Σ|x_i y_i|
    left_side = np.sum(np.abs(x * y))
    
    # 计算不等式右边：||x||_p · ||y||_q
    right_side = p_norm(x, p) * p_norm(y, q)
    
    # 验证不等式：左边 ≤ 右边 + 容差
    return left_side <= right_side + tol


def demonstrate_cauchy_schwarz(x: np.ndarray, y: np.ndarray) -> bool:
    """
    演示柯西-施瓦茨不等式（霍尔德不等式的特例，p=q=2）
    
    不等式形式：(Σx_i y_i)^2 ≤ (Σx_i^2)(Σy_i^2)
    
    参数:
    x: np.ndarray - 第一个向量
    y: np.ndarray - 第二个向量
    
    返回:
    bool - 如果不等式成立，返回True；否则返回False
    """
    print("柯西-施瓦茨不等式演示")
    print("=====================")
    
    # 计算点积的平方
    dot_product_sq = (np.dot(x, y))**2
    
    # 计算向量2-范数的平方的乘积
    norm_product_sq = (np.sum(x**2)) * (np.sum(y**2))
    
    print(f"(x·y)² = {dot_product_sq:.4f}")
    print(f"||x||₂² · ||y||₂² = {norm_product_sq:.4f}")
    print(f"是否满足柯西-施瓦茨不等式：{dot_product_sq <= norm_product_sq + 1e-8}")
    
    # 同时验证这也是霍尔德不等式的特例（p=q=2）
    is_holder = verify_holder_inequality(x, y, p=2, q=2)
    print(f"作为霍尔德不等式特例是否成立：{is_holder}")
    print()
    
    return dot_product_sq <= norm_product_sq + 1e-8


def demonstrate_holder_variations(x: np.ndarray, y: np.ndarray) -> None:
    """
    使用不同的p和q值演示霍尔德不等式
    """
    print("不同p和q值的霍尔德不等式演示")
    print("=============================")
    
    # 测试不同的p值（对应的q值 = 1/(1-1/p)）
    p_values = [1.5, 2, 3, 4, 5, np.inf]
    
    for p in p_values:
        if p == np.inf:
            q = 1.0
        else:
            q = 1 / (1 - 1/p)
        
        print(f"\np = {p}, q = {q}")
        
        # 计算左边
        left_side = np.sum(np.abs(x * y))
        
        # 计算右边
        right_side = p_norm(x, p) * p_norm(y, q)
        
        print(f"左边：Σ|x_i y_i| = {left_side:.6f}")
        print(f"右边：||x||_p · ||y||_q = {right_side:.6f}")
        print(f"是否满足不等式：{left_side <= right_side + 1e-8}")
        print(f"比率（左边/右边）：{left_side/right_side:.10f}")


if __name__ == "__main__":
    print("知识点8：霍尔德（Hölder）不等式")
    print("===============================")
    print("定义：对任意向量x, y ∈ R^n，当1 ≤ p ≤ ∞且1 ≤ q ≤ ∞满足1/p + 1/q = 1时，有：")
    print("Σ_{i=1}^n |x_i y_i| ≤ ||x||_p · ||y||_q")
    print()
    print("特殊情况：")
    print("- 当p=q=2时，即为柯西-施瓦茨不等式")
    print("- 当p=1时，q=∞，不等式变为Σ|x_i y_i| ≤ ||x||₁ · ||y||_∞")
    print()
    
    # 创建示例向量
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 10])
    
    print(f"示例向量x = {x}")
    print(f"示例向量y = {y}")
    print()
    
    # 演示柯西-施瓦茨不等式（p=q=2）
    demonstrate_cauchy_schwarz(x, y)
    
    # 演示不同p和q值的霍尔德不等式
    demonstrate_holder_variations(x, y)
    
    # 测试边界情况：p=1，q=∞
    print("\n边界情况演示：p=1, q=∞")
    print("=====================")
    left_side = np.sum(np.abs(x * y))
    right_side = p_norm(x, 1) * p_norm(y, np.inf)
    print(f"左边：Σ|x_i y_i| = {left_side:.6f}")
    print(f"右边：||x||₁ · ||y||_∞ = {right_side:.6f}")
    print(f"是否满足不等式：{left_side <= right_side + 1e-8}")
    print(f"比率（左边/右边）：{left_side/right_side:.10f}")
    
    # 验证对于随机向量的情况
    print("\n随机向量测试")
    print("============")
    num_tests = 100
    p = 3.0
    q = 1 / (1 - 1/p)
    
    passed_tests = 0
    for i in range(num_tests):
        # 生成随机向量
        rand_x = np.random.randn(10)  # 10维随机向量
        rand_y = np.random.randn(10)
        
        # 验证霍尔德不等式
        if verify_holder_inequality(rand_x, rand_y, p, q):
            passed_tests += 1
    
    print(f"对{p}和{q}的情况下，{num_tests}次随机向量测试中通过了{passed_tests}次")
    print(f"通过率：{passed_tests/num_tests:.2%}")
