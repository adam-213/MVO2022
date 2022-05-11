from constants import *


def basil(f, df, x0, eps=1e-8, n=10000):
    """Barzilai & Borwein method of semi-optimal step based on https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf"""
    x_k = x0.copy()
    x_k_1 = eps
    g_k_1 = eps
    gradients = []
    i = 0
    for i in tqdm(range(n)):
        if np.isnan(x_k[0]):
            return None, None
        if i != 0 and (x_k == x_pred).all():
            return None, None
        if np.linalg.norm(df(x_k)) < eps:
            break
        gradients.append(df(x_k))
        g_k = df(x_k)
        delta_x = x_k - x_k_1
        delta_g = g_k - g_k_1
        alpha_k = np.dot(delta_x, delta_x) / np.dot(delta_x, delta_g)
        # s = alpha_k * np.eye(x_k.shape[0])
        x_k_1 = x_k
        g_k_1 = g_k
        x_pred = x_k
        x_k = x_k - alpha_k * g_k

    return x_k, i, gradients


def basil_file():
    if generate:
        # TODO FILL
        bs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0.1, 0.9, 1.1]]
        barbor_results = []
        for b in bs:
            b = np.array(b)
            res = basil(f, df, b, eps, n)
            barbor_results.append((res, b))

        with open(root.joinpath('Data/BarBor.pickle'), 'wb') as c:
            pickle.dump(barbor_results, c)

    else:
        with open(root.joinpath('Data/BarBor.pickle'), 'rb') as c:
            barbor_results = pickle.load(c)
    return barbor_results


def barbor_result_file():
    def basil(f, df, x0, eps=1e-8, n=10000):
        """Barzilai & Borwein method of semi-optimal step based on https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf"""
        x_k = x0.copy()
        x_k_1 = eps
        g_k_1 = eps
        xis = []
        i = 0
        for i in tqdm(range(n)):
            if np.isnan(x_k[0]):
                break
            if i != 0 and (x_k == x_pred).all():
                break
            if np.linalg.norm(df(x_k)) < eps:
                break
            xis.append(x_k)
            g_k = df(x_k)
            delta_x = x_k - x_k_1
            delta_g = g_k - g_k_1
            alpha_k = np.dot(delta_x, delta_x) / np.dot(delta_x, delta_g)
            # s = alpha_k * np.eye(x_k.shape[0])
            x_k_1 = x_k
            g_k_1 = g_k
            x_pred = x_k
            x_k = x_k - alpha_k * g_k

        return xis, x0

    if generate:
        # TODO FILL
        bs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0.1, 0.9, 1.1]]
        barbor_results = []
        for b in bs:
            b = np.array(b)
            res = basil(f, df, b, eps, n)
            barbor_results.append((res, b))

        with open(root.joinpath('Data/BarBor_results.pickle'), 'wb') as c:
            pickle.dump(barbor_results, c)

    else:
        with open(root.joinpath('Data/BarBor_results.pickle'), 'rb') as c:
            barbor_results = pickle.load(c)
    return barbor_results
