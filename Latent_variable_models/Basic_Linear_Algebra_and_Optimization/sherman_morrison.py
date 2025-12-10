import numpy as np


def sherman_morrison(A: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    使用Sherman-Morrison公式计算(A + u v^⊤)的逆矩阵
    
    参数:
    A: np.ndarray - 可逆的n×n矩阵
    u: np.ndarray - n维列向量
    v: np.ndarray - n维行向量
    
    返回:
    np.ndarray - (A + u v^⊤)的逆矩阵
    
    抛出:
    ValueError - 如果A不可逆或1 + v^⊤ A^{-1} u = 0
    """
    # 确保A是方阵
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("矩阵A必须是方阵")
    
    # 确保u和v是正确维度的向量
    if u.shape != (n, 1) and u.shape != (n,):
        raise ValueError("向量u必须是n维向量")
    if v.shape != (1, n) and v.shape != (n,):
        raise ValueError("向量v必须是n维向量")
    
    # 确保A可逆
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("矩阵A不可逆")
    
    # 计算标量部分
    v_Ainv_u = v @ A_inv @ u
    denominator = 1 + v_Ainv_u
    
    if denominator == 0:
        raise ValueError("Sherman-Morrison公式的分母为零，(A + u v^⊤)不可逆")
    
    # 计算(A + u v^⊤)的逆矩阵
    term = (A_inv @ u.reshape(n, 1)) @ (v.reshape(1, n) @ A_inv)
    A_plus_uvT_inv = A_inv - term / denominator
    
    return A_plus_uvT_inv


def woodbury_identity(A: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    使用Woodbury矩阵恒等式计算(A + U V^⊤)的逆矩阵
    
    参数:
    A: np.ndarray - 可逆的n×n矩阵
    U: np.ndarray - n×k矩阵
    V: np.ndarray - n×k矩阵
    
    返回:
    np.ndarray - (A + U V^⊤)的逆矩阵
    
    抛出:
    ValueError - 如果A不可逆或(I_k + V^⊤ A^{-1} U)不可逆
    """
    # 确保A是方阵
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("矩阵A必须是方阵")
    
    # 确保U和V是正确维度的矩阵
    if U.shape[0] != n or V.shape[0] != n:
        raise ValueError("矩阵U和V必须是n×k矩阵")
    
    # 确保A可逆
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("矩阵A不可逆")
    
    # 计算中间矩阵
    k = U.shape[1]
    middle = np.eye(k) + V.T @ A_inv @ U
    
    # 确保中间矩阵可逆
    try:
        middle_inv = np.linalg.inv(middle)
    except np.linalg.LinAlgError:
        raise ValueError("Woodbury矩阵恒等式中的中间矩阵不可逆，(A + U V^⊤)不可逆")
    
    # 计算(A + U V^⊤)的逆矩阵
    A_plus_UVT_inv = A_inv - A_inv @ U @ middle_inv @ V.T @ A_inv
    
    return A_plus_UVT_inv


if __name__ == "__main__":
    print("知识点3：Sherman-Morrison公式及其推广")
    print("========================================")
    print("Sherman-Morrison公式：")
    print("对于可逆矩阵A∈R^{n×n}和向量u,v∈R^n，")
    print("(A + u v^⊤)^{-1} = A^{-1} - (A^{-1} u v^⊤ A^{-1}) / (1 + v^⊤ A^{-1} u)")
    print()
    print("Woodbury矩阵恒等式（推广形式）：")
    print("对于矩阵U,V∈R^{n×k}，")
    print("(A + U V^⊤)^{-1} = A^{-1} - A^{-1} U (I_k + V^⊤ A^{-1} U)^{-1} V^⊤ A^{-1}")
    print()
    
    # 示例1：使用Sherman-Morrison公式
    print("示例1：使用Sherman-Morrison公式")
    n = 4
    A = np.eye(n)  # 单位矩阵，显然可逆
    u = np.random.randn(n, 1)
    v = np.random.randn(1, n)
    
    # 使用Sherman-Morrison公式计算逆矩阵
    A_plus_uvT_inv_sm = sherman_morrison(A, u, v)
    
    # 直接计算逆矩阵（用于验证）
    A_plus_uvT = A + u @ v
    A_plus_uvT_inv_direct = np.linalg.inv(A_plus_uvT)
    
    print("验证Sherman-Morrison公式的结果：")
    print(f"两种方法计算的逆矩阵是否接近：{np.allclose(A_plus_uvT_inv_sm, A_plus_uvT_inv_direct)}")
    print(f"最大绝对误差：{np.max(np.abs(A_plus_uvT_inv_sm - A_plus_uvT_inv_direct)):.10f}")
    print()
    
    # 示例2：使用Woodbury矩阵恒等式
    print("示例2：使用Woodbury矩阵恒等式")
    k = 2
    U = np.random.randn(n, k)
    V = np.random.randn(n, k)
    
    # 使用Woodbury矩阵恒等式计算逆矩阵
    A_plus_UVT_inv_wi = woodbury_identity(A, U, V)
    
    # 直接计算逆矩阵（用于验证）
    A_plus_UVT = A + U @ V.T
    A_plus_UVT_inv_direct = np.linalg.inv(A_plus_UVT)
    
    print("验证Woodbury矩阵恒等式的结果：")
    print(f"两种方法计算的逆矩阵是否接近：{np.allclose(A_plus_UVT_inv_wi, A_plus_UVT_inv_direct)}")
    print(f"最大绝对误差：{np.max(np.abs(A_plus_UVT_inv_wi - A_plus_UVT_inv_direct)):.10f}")
    print()
    
    # 示例3：比较计算效率（对于大型矩阵）
    print("示例3：比较计算效率")
    n_large = 1000
    k_large = 10
    
    A_large = np.eye(n_large)
    U_large = np.random.randn(n_large, k_large)
    V_large = np.random.randn(n_large, k_large)
    
    # 使用Woodbury矩阵恒等式
    import time
    start_time = time.time()
    A_plus_UVT_inv_wi = woodbury_identity(A_large, U_large, V_large)
    wi_time = time.time() - start_time
    
    # 直接计算逆矩阵（对于大型矩阵可能很慢）
    start_time = time.time()
    A_plus_UVT = A_large + U_large @ V_large.T
    A_plus_UVT_inv_direct = np.linalg.inv(A_plus_UVT)
    direct_time = time.time() - start_time
    
    print(f"矩阵大小：{n_large}×{n_large}")
    print(f"Woodbury矩阵恒等式计算时间：{wi_time:.6f}秒")
    print(f"直接计算逆矩阵时间：{direct_time:.6f}秒")
    print(f"速度提升：{direct_time / wi_time:.2f}倍")
    print(f"结果是否一致：{np.allclose(A_plus_UVT_inv_wi, A_plus_UVT_inv_direct)}")
