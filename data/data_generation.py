# install packages before running this code:
# e.g., pip3 install numpy seaborn pandas
import numpy as np
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    np.random.seed(0)

    label_colors = {0: '#fb694a', 1: '#6aaed6'}

    penguins = sns.load_dataset('penguins')
    penguins = penguins[['species', 'body_mass_g',
                         'flipper_length_mm']].dropna()

    y = np.array(penguins['species'])
    y[(y != 'Adelie')] = 1
    y[y == 'Adelie'] = 0

    X = np.array(penguins[['body_mass_g', 'flipper_length_mm']])
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X[y == 1, :] = X[y == 1, :] + 0.22
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    noise = np.random.normal(0, .1, X.shape)
    X2 = X + noise
    X = np.vstack((X, X2, X, X))
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = np.hstack((y, y, y, y))

    n_random_points = 1200
    random_points = np.random.rand(n_random_points, 2)
    random_labels = np.random.randint(2, size=n_random_points)

    X = np.vstack((X, random_points))
    y = np.hstack((y, random_labels))

    df = pd.DataFrame(y, columns=['label'])
    df['x_pos'] = X[:, 0]
    df['y_pos'] = X[:, 1]
    df.to_csv('./sample_data.csv', index=False)