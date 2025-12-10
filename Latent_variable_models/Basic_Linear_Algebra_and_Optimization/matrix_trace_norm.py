import numpy as np


def matrix_trace(matrix: np.ndarray) -> float:
    """
    计算矩阵的迹
    
    参数:
    matrix: np.ndarray - 输入矩阵
    
    返回:
    float - 矩阵的迹（对角线元素之和）
    """
    return np.trace(matrix)


def frobenius_norm(matrix: np.ndarray) -> float:
    """
    计算矩阵的弗罗贝尼乌斯范数
    
    参数:
    matrix: np.ndarray - 输入矩阵
    
    返回:
    float - 矩阵的弗罗贝尼乌斯范数
    """
    return np.linalg.norm(matrix, ord='fro')


def spectral_norm(matrix: np.ndarray) -> float:
    """
    计算矩阵的谱范数（2→2范数）
    
    参数:
    matrix: np.ndarray - 输入矩阵
    
    返回:
    float - 矩阵的谱范数
    """
    return np.linalg.norm(matrix, ord=2)


def demonstrate_trace_properties():
    """
    演示迹的基本性质
    """
    print("迹的基本性质演示")
    print("==================")
    
    # 创建随机矩阵
    n = 4
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    print(f"矩阵A的迹: {matrix_trace(A)}")
    print(f"矩阵B的迹: {matrix_trace(B)}")
    
    # 迹的线性性质：tr(A + B) = tr(A) + tr(B)
    trace_A_plus_B = matrix_trace(A + B)
    trace_A_plus_trace_B = matrix_trace(A) + matrix_trace(B)
    print(f"\n迹的线性性质：tr(A + B) = tr(A) + tr(B)")
    print(f"tr(A + B): {trace_A_plus_B}")
    print(f"tr(A) + tr(B): {trace_A_plus_trace_B}")
    print(f"是否相等: {np.isclose(trace_A_plus_B, trace_A_plus_trace_B)}")
    
    # 迹的交换性：tr(AB) = tr(BA)
    trace_AB = matrix_trace(A @ B)
    trace_BA = matrix_trace(B @ A)
    print(f"\n迹的交换性：tr(AB) = tr(BA)")
    print(f"tr(AB): {trace_AB}")
    print(f"tr(BA): {trace_BA}")
    print(f"是否相等: {np.isclose(trace_AB, trace_BA)}")
    
    # 迹与二次型的关系：x^T A x = tr(A x x^T)
    x = np.random.randn(n)
    quadratic_form = x.T @ A @ x
    trace_A_xxT = matrix_trace(A @ x @ x.T)
    print(f"\n迹与二次型的关系：x^T A x = tr(A x x^T)")
    print(f"x^T A x: {quadratic_form}")
    print(f"tr(A x x^T): {trace_A_xxT}")
    print(f"是否相等: {np.isclose(quadratic_form, trace_A_xxT)}")


if __name__ == "__main__":
    print("知识点2：矩阵的迹、范数和基本性质")
    print("====================================")
    print("定义：")
    print("- 迹：tr(A) := Σ_i a_ii")
    print("- 弗罗贝尼乌斯范数：||A||_F² := tr(A^⊤ A) = √(Σ_{i,j} a_ij²)")
    print("- 谱范数：||A||_{2→2} = sup_{x: ||x||_2=1} ||A x||_2")
    print()
    
    # 创建示例矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    print("示例矩阵A：")
    print(A)
    print()
    
    # 计算迹
    print(f"矩阵A的迹：{matrix_trace(A)}")
    
    # 计算弗罗贝尼乌斯范数
    print(f"矩阵A的弗罗贝尼乌斯范数：{frobenius_norm(A):.4f}")
    
    # 计算谱范数
    print(f"矩阵A的谱范数：{spectral_norm(A):.4f}")
    print()
    
    # 演示迹的性质
    demonstrate_trace_properties()
