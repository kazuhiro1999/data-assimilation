import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.colors import LinearSegmentedColormap

# 図のスタイル設定
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = "MS Gothic"

# ランダムシードを設定（再現性のため）
np.random.seed(42)

# 2次元グリッドを作成
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 2つのガウス分布のパラメータを定義
mean1 = np.array([0, 0])  # 平均値1
cov1 = np.array([[4.0, 2.0], [2.0, 4.0]])  # 共分散行列1
weight1 = 0.7  # 重み1

mean2 = np.array([4, 2])  # 平均値2
cov2 = np.array([[3, -1], [-1, 3]])  # 共分散行列2
weight2 = 0.3  # 重み2

# 各ガウス分布の確率密度関数を計算
rv1 = multivariate_normal(mean1, cov1)
rv2 = multivariate_normal(mean2, cov2)

# 混合分布の確率密度関数を計算
z = weight1 * rv1.pdf(pos) + weight2 * rv2.pdf(pos)

# カスタムのカラーマップを作成
colors = [(0.98, 0.98, 1), (0, 0.3, 0.8)]
cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)

# ヒートマップの描画
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, z, 50, cmap=cmap)
#plt.colorbar(contour, label='確率密度')


# 等高線を追加
plt.contour(X, Y, z, 7, colors='white', alpha=0.3, linewidths=0.8)

plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                labelbottom=False, left=False, right=False, labelleft=False)

plt.savefig('gaussian_mixture_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 混合分布からサンプリング
def sample_gaussian_mixture(n_samples, means, covs, weights):
    # 各ガウス分布から何個サンプリングするかを決定
    n_samples_per_gaussian = np.random.multinomial(n_samples, weights)
    samples = []
    
    # 各ガウス分布からサンプリング
    for i in range(len(means)):
        if n_samples_per_gaussian[i] > 0:
            samples.append(np.random.multivariate_normal(means[i], covs[i], n_samples_per_gaussian[i]))
    
    # サンプルを結合
    return np.vstack(samples)

# サンプリングを実行
means = [mean1, mean2]
covs = [cov1, cov2]
weights = [weight1, weight2]
samples = sample_gaussian_mixture(300, means, covs, weights)

# サンプルの散布図
plt.figure(figsize=(10, 8))

# ヒートマップを背景に表示（薄く）
plt.contourf(X, Y, z, 50, cmap=cmap, alpha=0.3)

# サンプル点をプロット
plt.scatter(samples[:, 0], samples[:, 1], c='darkblue', s=50, alpha=0.7, edgecolor='none', label='パーティクル')



plt.grid(alpha=0.3)

plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                labelbottom=False, left=False, right=False, labelleft=False)

plt.savefig('gaussian_mixture_samples.png', dpi=300, bbox_inches='tight')
plt.show()