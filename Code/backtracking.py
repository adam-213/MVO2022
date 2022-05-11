from constants import *


def grad_backtracking(f, df, x0, eps=1e-2, n=1000000, alpha=0.1, delta=0.5, c=1):
    """semi-optimal step gradient method with backtracking for step length"""
    xk = x0
    i = 0
    for i in tqdm(range(n)):
        if np.isnan(xk[0]):
            return None, None
        if i != 0 and (xk == x_pred).all():
            return None, None
        if np.linalg.norm(df(xk)) < eps:
            break
        s = -df(xk)
        while np.linalg.norm(f(xk + c * s)) > np.linalg.norm(f(xk) + c * alpha * s.T @ df(xk)):
            c = delta * c
        x_pred = xk
        xk = xk + c * s

    return xk, i


def backtracking_file():
    if generate:
        alphas = [0.1, 0.51]
        deltas = [0.2, 0.9]
        cs = [2, 1, 1e-1]
        bs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0.1, 0.9, 1.1]]
        backtracking_results = []

        for alpha in alphas:
            for delta in deltas:
                for c in cs:
                    for b in bs:
                        b = np.array(b)
                        res = grad_backtracking(f, df, b, eps, n, alpha, delta, int(c))
                        backtracking_results.append((res, b, c, alpha, delta))

        with open(root.joinpath('Data/backtracking.pickle'), "wb") as c:
            # this file may not be correct
            pickle.dump(backtracking_results, c)
    else:
        with open(root.joinpath('Data/backtracking.pickle'), 'rb') as c:
            backtracking_results = pickle.load(c)

    return backtracking_results


def back_result_file():
    import ray

    @ray.remote
    def grad_backtracking(f, df, x0, eps=1e-2, n=1000000, alpha=0.1, delta=0.5, c=1):
        """semi-optimal step gradient method with backtracking for step length"""
        xis = []

        xk = x0
        i = 0
        for i in range(n):
            if np.isnan(xk[0]):
                break
            if i != 0 and (xk == x_pred).all():
                break
            if np.linalg.norm(df(xk)) < eps:
                break
            if i % 100 == 0:
                xis.append(xk)
            if i % 10000 == 0:
                print(i)
            s = -df(xk)
            while np.linalg.norm(f(xk + c * s)) > np.linalg.norm(f(xk) + c * alpha * s.T @ df(xk)):
                c = delta * c
            x_pred = xk
            xk = xk + c * s

        return xis, x0, c, alpha, delta

    if generate:
        ray.init()
        alphas = [0.1, 0.51]
        deltas = [0.2, 0.9]
        cs = [2, 1, 1e-1]
        bs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0.1, 0.9, 1.1]]
        result_ids = []

        for alpha in alphas:
            for delta in deltas:
                for c in cs:
                    for b in bs:
                        b = np.array(b)
                        result_ids.append(grad_backtracking.remote(f, df, b, eps, n, alpha, delta, int(c)))
        results = ray.get(result_ids)
        with open(root.joinpath('Data/backtracking_result.pickle'), "wb") as c:
            pickle.dump(results, c)
    else:
        with open(root.joinpath('Data/backtracking_result.pickle'), 'rb') as c:
            results = pickle.load(c)
    if generate:
        ray.shutdown()
    return results

