# install coloraide by (pip3 install coloraide)
from coloraide import Color
from scipy.stats import gaussian_kde
from matplotlib import cm


def density_and_class_ratio(X, y, normalize_density=True):
    '''
	inputs
	-----
	X: 2D positions (shape: (n_samples, 2))
	y: 2-class labels. must be 0 or 1 (shape: (n_samples, ))
	normalize_density: whether scaling density within 0-1

	outputs
	----
	density: density (shape: (n_samples, ))
	class1_ratio: ratio of class1/(class0 + class1) (shape: (n_samples, ))
	'''
    density = gaussian_kde(X.T)(X.T)
    order = density.argsort()

    kde0 = gaussian_kde(X[y == 0].T)
    kde1 = gaussian_kde(X[y == 1].T)

    n0 = X[y == 0, :].shape[0]
    n1 = X[y == 0, :].shape[0]
    ratio_estimation = lambda x: n1 * kde1(x) / (n0 * kde0(x) + n1 * kde1(x))

    class1_ratio = ratio_estimation(X.T)

    if normalize_density:
        density = (density - density.min()) / (density.max() - density.min())

    return density, class1_ratio


def gen_polar_colormap(cmap0=cm.Reds,
                       cmap1=cm.Blues,
                       interpolation_space='lch',
                       interpolation_hue_selection='shorter'):
    '''
	inputs
	-----
	cmap0: matplotlib colormap corresponding to class0
	cmap1: matplotlib colormap corresponding to class1
	interpolation_space: color space used for interpolation. check coloraide.Color.interpolate
	interpolation_hue_selection: a way for hue interpolation. check coloraide.Color.interpolate

	outputs
	-----
	polor_colormap: function to generate color from radius and ratio of class1
		
        inputs
		-----
		radius: decides location of each colormap
		ratio1: decides ratio of how much the interpolation inclines to cmap1

		outputs
		-----
		r, g, b, a: colors and alphs in ranges of 0-1
	'''

    def polor_colormap(radius, ratio1):
        r0, g0, b0, a0 = cmap0(radius)
        r1, g1, b1, a1 = cmap1(radius)

        ci = Color.interpolate(
            [Color('srgb', [r0, g0, b0]),
             Color('srgb', [r1, g1, b1])],
            space=interpolation_space,
            out_space='srgb',
            hue=interpolation_hue_selection)
        r, g, b, _ = ci(ratio1)

        a = a0 + ratio1 * (a1 - a0)
        r, g, b, a = [min(e, 1) for e in [r, g, b, a]]
        r, g, b, a = [max(0, e) for e in [r, g, b, a]]

        return r, g, b, a

    return polor_colormap


def polar_colormap(radius, ratio1):
    '''
    Default polar colormap used in the paper below:
    Lu et al., "Visual Analytics of Multivariate Networks with Representation Learning and Composite Variable Construction", Visual Informatics, 2023 (forthcoming).
    
    inputs
    -----
    radius: decides color lightness
    ratio1: decides ratio of how much color hue inclines from red to blue

    outputs
    -----
    r, g, b, a: colors and alphs in ranges of 0-1
    '''
    return gen_polar_colormap()(radius, ratio1)


def gen_checkviz_colormap(values1, values2):
    # implementation of bivariate colormap introduced in:
    # CheckViz: Sanity Check and Topological Clues for Linear and Non-Linear Mappings
    # https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2010.01835.x

    # to implement this function, referred to:
    # https://github.com/hj-n/snc-reliability-map/blob/master/src/components/Checkviz.js

    import numpy as np

    max1 = np.max(values1)
    min1 = np.min(values1)
    max2 = np.max(values2)
    min2 = np.min(values2)

    def return_color(val1, val2):
        v1 = (val1 - min1) / (max1 - min1)
        v2 = (val2 - min2) / (max2 - min2)
        va = min(1, max(v1 - v2, -1))
        vb = min(1, max(v2 - v1, -1))
        vl = min(1, (max(1 - (v1 + v2) / 2, 0)))

        scale = 1.3
        a = va * 30 * scale
        b = vb * 20 * scale
        l = vl**1.5145 * 100

        r, g, b, a = Color('lab', [l, a, b]).convert('srgb')
        # coloraide returns negative value sometimes
        r, g, b, a = [max(x, 0) for x in [r, g, b, a]]

        return r, g, b, a

    return return_color


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('./data/sample_data.csv')
    y = np.array(df['label'])
    X = np.array(df[['x_pos', 'y_pos']])

    # compute density and class 1 ratio
    density, class1_ratio = density_and_class_ratio(X, y)
    # polar colormap
    pcmap = polar_colormap

    # checkviz colormap just for comparison with polor colormap
    cvmap = gen_checkviz_colormap(density, class1_ratio)

    fig, axs = plt.subplots(ncols=4, figsize=(12.5, 3.5))

    # ordinary two-class colored scatteplot
    order = np.arange(X.shape[0])
    np.random.shuffle(order)
    colors = np.array(['#fb694a'] * X.shape[0])
    colors[y == 1] = '#6aaed6'

    axs[0].scatter(X[order, 0],
                   X[order, 1],
                   c=colors,
                   edgecolors='#888888',
                   lw=0.2)
    axs[0].set_title('two-class scatter', fontsize=9)

    # ordinary single class density scatterplot
    order = np.argsort(density)
    axs[1].scatter(X[order, 0],
                   X[order, 1],
                   c=density[order],
                   edgecolors='#888888',
                   lw=0.2,
                   zorder=10,
                   cmap='Greens')
    axs[1].set_title('density scatter', fontsize=9)

    # two-class densiy scatteplot but using existing colormap (checkcviz colormap)
    colors = np.array([
        cvmap(radius, ratio1) for radius, ratio1 in zip(density, class1_ratio)
    ])
    order = np.argsort(density)
    axs[2].scatter(X[order, 0],
                   X[order, 1],
                   c=colors[order],
                   edgecolors='#888888',
                   lw=0.2)
    axs[2].set_title('two-class density scatter (w/ CheckViz colormap)',
                     fontsize=9)

    # two-class densiy scatteplot but using polar colormap
    colors = np.array([
        pcmap(radius, ratio1) for radius, ratio1 in zip(density, class1_ratio)
    ])
    order = np.argsort(density)
    axs[3].scatter(X[order, 0],
                   X[order, 1],
                   c=colors[order],
                   edgecolors='#888888',
                   lw=0.2)
    axs[3].set_title('two-class density scatter (w/ new colormap)', fontsize=9)

    for ax in axs:
        ax.set(xticks=[])
        ax.set(yticks=[])
        ax.set(xlabel=None)
        ax.set(ylabel=None)

    plt.tight_layout()
    plt.show()

    # plot color legend
    from matplotlib.colors import ListedColormap

    n = 200
    angle = 60

    fig, _ = plt.subplots(1, 1)
    c_density, c_ratio = np.meshgrid(np.linspace(0, 1, n),
                                     np.linspace(0, 1, n))
    theta = c_ratio * (angle / 180) * np.pi - (angle / 360) * np.pi

    # make 1D colormap used to map location to a polar colormap
    cmap_colors = []
    for i in range(n):
        for j in range(n):
            cmap_colors.append(pcmap(c_density[i, j], c_ratio[i, j]))
    newcmp = ListedColormap(cmap_colors)

    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z[i, j] = n * i + j

    ax = plt.subplot(projection='polar')
    ax.grid(False)
    plt.pcolormesh(theta, c_density, z, cmap=newcmp, shading='auto')
    plt.xticks([])
    plt.yticks([])
    ax.axis('off')
    # plt.savefig('./images/polar_colormap.png', transparent=True, dpi=300)
    plt.show()
