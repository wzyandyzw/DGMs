import numpy as np
import matplotlib.pyplot as plt
# 设置Matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def is_convex(f, x, y, alpha: float = 0.5, tol: float = 1e-8) -> bool:
    """
    检查函数在两点之间是否满足凸函数的定义
    
    参数:
    f: callable - 单变量函数f(x)
    x: float - 第一个点
    y: float - 第二个点
    alpha: float - [0, 1]之间的参数
    tol: float - 容差
    
    返回:
    bool - 如果满足凸函数定义，返回True；否则返回False
    """
    # 计算凸组合点
    z = alpha * x + (1 - alpha) * y
    
    # 计算函数值
    f_z = f(z)
    
    # 计算函数值的凸组合
    convex_comb = alpha * f(x) + (1 - alpha) * f(y)
    
    # 检查是否满足f(αx + (1-α)y) ≤ αf(x) + (1-α)f(y)
    return f_z <= convex_comb + tol


def check_convexity(f, domain: tuple = (-10, 10), num_tests: int = 100) -> bool:
    """
    在定义域内随机测试函数是否为凸函数
    
    参数:
    f: callable - 单变量函数f(x)
    domain: tuple - 定义域区间 (min_x, max_x)
    num_tests: int - 测试次数
    
    返回:
    bool - 如果所有测试都通过，返回True；否则返回False
    """
    min_x, max_x = domain
    
    for _ in range(num_tests):
        # 随机生成两个点x和y
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_x, max_x)
        
        # 随机生成alpha
        alpha = np.random.uniform(0, 1)
        
        # 检查凸函数定义
        if not is_convex(f, x, y, alpha):
            return False
    
    return True


def visualize_convexity(f, domain: tuple = (-10, 10), title: str = "Convex Function"):
    """
    可视化函数及其凸性
    
    参数:
    f: callable - 单变量函数f(x)
    domain: tuple - 定义域区间 (min_x, max_x)
    title: str - 图像标题
    """
    # 生成函数图像的点
    x = np.linspace(domain[0], domain[1], 100)
    y = f(x)
    
    # 选择两个点来展示凸组合
    x1 = np.random.uniform(domain[0], domain[1])
    x2 = np.random.uniform(domain[0], domain[1])
    alpha = 0.3
    x3 = alpha * x1 + (1 - alpha) * x2
    
    # 计算对应点的函数值
    y1 = f(x1)
    y2 = f(x2)
    y3 = f(x3)
    convex_comb = alpha * y1 + (1 - alpha) * y2
    
    # 创建图像
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="f(x)", color="blue", linewidth=2)
    
    # 标记点和连线
    plt.scatter([x1, x2, x3], [y1, y2, y3], color=["red", "red", "green"], s=100)
    plt.plot([x1, x2], [y1, y2], color="red", linestyle="--", label="Line segment between f(x1) and f(x2)")
    plt.scatter(x3, convex_comb, color="purple", s=100, marker="x", label="αf(x1) + (1-α)f(x2)")
    plt.plot([x3, x3], [y3, convex_comb], color="green", linestyle="-", label="f(αx1 + (1-α)x2) vs αf(x1) + (1-α)f(x2)")
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("知识点5：凸函数")
    print("================")
    print("定义：如果对于所有x, y ∈ R^n和所有α ∈ [0, 1]，函数f : R^n → R满足")
    print("f(αx + (1 - α)y) ≤ αf(x) + (1 - α)f(y)")
    print("则称该函数为凸函数")
    print()
    
    # 示例1：二次函数f(x) = x²（凸函数）
    print("示例1：二次函数 f(x) = x²")
    f1 = lambda x: x ** 2
    print(f"是否为凸函数：{check_convexity(f1)}")
    
    # 示例2：指数函数f(x) = e^x（凸函数）
    print("\n示例2：指数函数 f(x) = e^x")
    f2 = lambda x: np.exp(x)
    print(f"是否为凸函数：{check_convexity(f2)}")
    
    # 示例3：对数函数f(x) = log(x)（凹函数）
    print("\n示例3：对数函数 f(x) = log(x)")
    f3 = lambda x: np.log(x)
    print(f"是否为凸函数：{check_convexity(f3, domain=(0.1, 10))}")
    print(f"其负函数 -f(x) = -log(x) 是否为凸函数：{check_convexity(lambda x: -np.log(x), domain=(0.1, 10))}")
    
    # 示例4：绝对值函数f(x) = |x|（凸函数）
    print("\n示例4：绝对值函数 f(x) = |x|")
    f4 = lambda x: np.abs(x)
    print(f"是否为凸函数：{check_convexity(f4)}")
    
    # 示例5：线性函数f(x) = ax + b（既是凸函数也是凹函数）
    print("\n示例5：线性函数 f(x) = 2x + 3")
    f5 = lambda x: 2 * x + 3
    print(f"是否为凸函数：{check_convexity(f5)}")
    
    # 可视化凸函数
    print("\n可视化凸函数示例：")
    print("展示函数f(x) = x²的凸性")
    visualize_convexity(f1, domain=(-5, 5), title="凸函数示例：f(x) = x²")
    
    print("展示函数f(x) = e^x的凸性")
    visualize_convexity(f2, domain=(-5, 5), title="凸函数示例：f(x) = e^x")
