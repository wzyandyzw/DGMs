import numpy as np


def perform_svd(matrix: np.ndarray, full_matrices: bool = True) -> tuple:
    """
    对矩阵进行奇异值分解(SVD)
    
    参数:
    matrix: np.ndarray - 输入矩阵，形状为(n, d)
    full_matrices: bool - 是否返回完整的U和V矩阵
    
    返回:
    tuple - (U, S, Vt)，其中：
        U: np.ndarray - 左奇异向量矩阵，形状为(n, n)或(n, r)，r为矩阵的秩
        S: np.ndarray - 奇异值向量，形状为(r,)
        Vt: np.ndarray - 右奇异向量矩阵的转置，形状为(d, d)或(r, d)
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=full_matrices)
    return U, S, Vt


def compact_svd(matrix: np.ndarray) -> tuple:
    """
    计算矩阵的紧凑SVD
    
    参数:
    matrix: np.ndarray - 输入矩阵
    
    返回:
    tuple - (U_r, S_r, Vt_r)，其中：
        U_r: np.ndarray - 左奇异向量矩阵，形状为(n, r)
        S_r: np.ndarray - 奇异值对角矩阵，形状为(r, r)
        Vt_r: np.ndarray - 右奇异向量矩阵的转置，形状为(r, d)
    """
    # 计算完整SVD
    U, S, Vt = perform_svd(matrix, full_matrices=False)
    
    # 构造奇异值对角矩阵
    r = S.shape[0]
    S_r = np.diag(S)
    
    return U, S_r, Vt


def matrix_approximation(matrix: np.ndarray, rank: int) -> np.ndarray:
    """
    使用SVD对矩阵进行低秩近似
    
    参数:
    matrix: np.ndarray - 输入矩阵
    rank: int - 近似矩阵的秩
    
    返回:
    np.ndarray - 低秩近似矩阵
    """
    # 计算完整SVD
    U, S, Vt = perform_svd(matrix, full_matrices=False)
    
    # 取前rank个奇异值和对应的奇异向量
    U_k = U[:, :rank]
    S_k = np.diag(S[:rank])
    Vt_k = Vt[:rank, :]
    
    # 重构低秩近似矩阵
    approx_matrix = U_k @ S_k @ Vt_k
    
    return approx_matrix


def compute_pseudoinverse(matrix: np.ndarray) -> np.ndarray:
    """
    使用SVD计算矩阵的伪逆
    
    参数:
    matrix: np.ndarray - 输入矩阵
    
    返回:
    np.ndarray - 矩阵的伪逆
    """
    # 计算完整SVD
    U, S, Vt = perform_svd(matrix, full_matrices=True)
    
    # 计算奇异值的伪逆（取倒数，零奇异值保持为零）
    S_inv = np.zeros((Vt.shape[0], U.shape[1]))
    S_inv[:len(S), :len(S)] = np.diag(1 / S)
    
    # 计算伪逆
    pseudoinverse = Vt.T @ S_inv @ U.T
    
    return pseudoinverse


if __name__ == "__main__":
    print("知识点4：奇异值分解（SVD）")
    print("===========================")
    print("定义：任意矩阵A∈R^{n×d}可分解为A = U Σ V^⊤")
    print("其中U和V是正交矩阵，Σ是矩形对角矩阵（对角线元素为非负值）")
    print()
    
    # 创建示例矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    
    print("示例矩阵A（形状：", A.shape, "）：")
    print(A)
    print()
    
    # 计算完整SVD
    print("1. 完整SVD分解：")
    U, S, Vt = perform_svd(A, full_matrices=True)
    
    print("U矩阵（左奇异向量）：")
    print(U.round(4))
    print(f"形状：{U.shape}")
    print()
    
    print("奇异值向量S：")
    print(S.round(4))
    print(f"形状：{S.shape}")
    print()
    
    print("V^⊤矩阵（右奇异向量的转置）：")
    print(Vt.round(4))
    print(f"形状：{Vt.shape}")
    print()
    
    # 验证SVD分解
    print("2. 验证SVD分解：")
    # 构造Σ矩阵
    n, d = A.shape
    Sigma = np.zeros((n, d))
    np.fill_diagonal(Sigma, S)
    
    A_reconstructed = U @ Sigma @ Vt
    print(f"原始矩阵和重构矩阵是否相等：{np.allclose(A, A_reconstructed)}")
    print(f"最大绝对误差：{np.max(np.abs(A - A_reconstructed)):.10f}")
    print()
    
    # 紧凑SVD
    print("3. 紧凑SVD分解：")
    U_r, S_r, Vt_r = compact_svd(A)
    
    print("U_r矩阵（左奇异向量，紧凑形式）：")
    print(U_r.round(4))
    print(f"形状：{U_r.shape}")
    print()
    
    print("S_r矩阵（奇异值对角矩阵，紧凑形式）：")
    print(S_r.round(4))
    print(f"形状：{S_r.shape}")
    print()
    
    print("Vt_r矩阵（右奇异向量的转置，紧凑形式）：")
    print(Vt_r.round(4))
    print(f"形状：{Vt_r.shape}")
    print()
    
    # 矩阵近似
    print("4. 矩阵低秩近似：")
    for rank in [1, 2]:
        approx_A = matrix_approximation(A, rank)
        print(f"\n秩-{rank}近似矩阵：")
        print(approx_A.round(4))
        error = np.linalg.norm(A - approx_A, 'fro')
        print(f"近似误差（弗罗贝尼乌斯范数）：{error:.6f}")
    print()
    
    # 计算伪逆
    print("5. 计算矩阵的伪逆：")
    A_pinv = compute_pseudoinverse(A)
    print("使用SVD计算的伪逆：")
    print(A_pinv.round(4))
    
    # 与numpy内置函数比较
    A_pinv_np = np.linalg.pinv(A)
    print(f"与numpy内置伪逆函数结果是否一致：{np.allclose(A_pinv, A_pinv_np)}")
    print()
    
    # 确定矩阵的秩
    print("6. 确定矩阵的秩：")
    print(f"矩阵A的秩：{len(S[S > 1e-10])}")
    print(f"numpy计算的秩：{np.linalg.matrix_rank(A)}")
