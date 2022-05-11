from constants import *

def cauchy(f, df, x0, eps=1e-2, n=1000000):
    for i in range(n):
        if np.linalg.norm(df(x0)) < eps:
            break
        s = -df(x0)
        mi = lambda x: (f(x0 + x * s))
        c = zlatyrez(mi, None, 0, 10000, 1e-8, n)
        x0 = x0 + c * s

    return x0,i


def zlatyrez(fun, df, a0, b0, eps, n):
    mi = 1 / ((1 + math.sqrt(5)) / 2)
    c1 = mi * a0 + (1 - mi) * b0
    c2 = (1 - mi) * a0 + mi * b0
    for i in range(n):
        if fun(c1) < fun(c2):
            b0 = c2
            c2 = c1
            c1 = mi * a0 + (1 - mi) * b0
        else:
            a0 = c1
            c1 = c2
            c2 = (1 - mi) * a0 + mi * b0
        if (b0 - a0) < eps:
            break

    if a0 == 0:
        return b0
    return a0


def cauchy_file():
    if generate:
        bs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [1, 7, -7], [0.1, 0.9, 1.1]]
        cauchy_results = []
        for b in bs:
            b = np.array(b)
            res = cauchy(f, df, b, eps, n)
            cauchy_results.append(res, b)

        with open(root.joinpath('Data/cauchy.pickle'), 'wb') as c:
            pickle.dump(cauchy_results, c)

    else:
        with open(root.joinpath("Data/cauchy.pickle"), 'rb') as c:
            cauchy_results = pickle.load(c)

    return cauchy_results


def cauchy_result_file():
    import ray

    @ray.remote
    def cauchy(f, df, x0, eps=1e-2, n=1000000):
        xbeg = x0.copy()
        xis = []
        for i in range(n):
            if np.linalg.norm(df(x0)) < eps:
                break
            if np.isnan(x0[0]):
                break
            if i != 0 and (x0 == x_pred).all():
                break
            if i % 100 == 0:
                xis.append(x0)
            if i % 1000 == 0:
                print(i)
            s = -df(x0)
            mi = lambda x: (f(x0 + x * s))
            c = zlatyrez(mi, None, 0, 10000, 1e-8, n)
            x_pred = x0
            x0 = x0 + c * s

        return xis, xbeg
    if generate:
        ray.init()
        bs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [1, 7, -7], [0.1, 0.9, 1.1]]
        result_ids = []
        for b in bs:
            b = np.array(b)
            result_ids.append(cauchy.remote(f, df, b, eps, n))

        results = ray.get(result_ids)
        with open(root.joinpath('Data/cauchy_results.pickle'), 'wb') as c:
            pickle.dump(results, c)
    else:
        with open(root.joinpath("Data/cauchy_results.pickle"), 'rb') as c:
            results = pickle.load(c)

    if generate:
        ray.shutdown()
    return results

