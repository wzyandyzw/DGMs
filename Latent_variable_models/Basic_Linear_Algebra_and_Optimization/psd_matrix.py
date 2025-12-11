import numpy as np


def is_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    检查一个矩阵是否为半正定矩阵(PSD)
    
    参数:
    matrix: np.ndarray - 输入矩阵，应为实对称矩阵
    tol: float - 容差，用于判断特征值是否非负
    
    返回:
    bool - 如果矩阵是半正定的，返回True；否则返回False
    """
    # 检查是否为实对称矩阵
    if not np.allclose(matrix, matrix.T):
        return False
    
    # 计算所有特征值
    eigenvalues, _ = np.linalg.eigh(matrix)
    
    # 检查所有特征值是否非负（考虑容差）
    return np.all(eigenvalues >= -tol)


def verify_psd_definition(matrix: np.ndarray, num_tests: int = 100, tol: float = 1e-8) -> bool:
    """
    通过定义验证矩阵是否为半正定矩阵：对任意向量x，检查x^T A x >= 0, 本判断无法做到数学上的严格成立，
    只能在数值上进行验证，通过生成多个随机向量进行测试。
    
    参数:
    matrix: np.ndarray - 输入矩阵
    num_tests: int - 测试的随机向量数量
    tol: float - 容差
    
    返回:
    bool - 如果通过所有测试，返回True；否则返回False
    """
    n = matrix.shape[0]
    
    for _ in range(num_tests):
        # 生成随机向量x
        x = np.random.randn(n)
        
        # 计算x^T A x
        quadratic_form = x.T @ matrix @ x
        
        # 检查是否非负
        if quadratic_form < -tol:
            return False
    
    return True


if __name__ == "__main__":
    print("知识点1：半正定矩阵")
    print("======================")
    print("定义：实对称矩阵A∈R^{n×n}是半正定（PSD）的，当且仅当对所有x∈R^n，都有x^⊤A x≥0")
    print("等价条件：所有特征值均非负的实对称矩阵是半正定矩阵")
    print()
    
    # 创建一个半正定矩阵的示例
    print("示例1：创建半正定矩阵（通过特征值分解）")
    n = 3
    eigenvalues = np.array([2.0, 1.0, 0.5])  # 所有特征值非负
    V = np.random.randn(n, n)
    Q, _ = np.linalg.qr(V)  # 正交矩阵
    A = Q @ np.diag(eigenvalues) @ Q.T  # 半正定矩阵
    
    print("矩阵A:")
    print(A)
    print()
    
    # 验证是否为半正定矩阵
    print("验证矩阵A是否为半正定矩阵：")
    print(f"1. 检查特征值是否非负：{is_psd(A)}")
    print(f"2. 通过定义验证：{verify_psd_definition(A)}")
    print()
    
    # 创建一个非半正定矩阵的示例
    print("示例2：创建非半正定矩阵")
    B = A.copy()
    B[0, 0] -= 3.0  # 使一个特征值变为负数
    
    print("矩阵B:")
    print(B)
    print()
    
    print("验证矩阵B是否为半正定矩阵：")
    print(f"1. 检查特征值是否非负：{is_psd(B)}")
    print(f"2. 通过定义验证：{verify_psd_definition(B)}")
