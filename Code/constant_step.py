from constants import *
def grad_const(f, df, x0, eps=1e-2, n=1000000, c=1e-5):
    """Gradient method with constant step"""
    xk = x0
    i = 0

    for i in tqdm(range(n)):
        if np.isnan(xk[0]):
            return None, None
        if i != 0 and (xk == x_pred).all():
            return None, None
        if np.linalg.norm(df(xk)) < eps:
            break
        x_pred = xk
        sk = -df(xk)
        xk = xk + (c * sk)

    return xk, i


def grad_const_file():
    if generate:
        cs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  # step length
        bs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [1, 7, -7], [0.1, 0.9, 1.1]]  # starting point

        const_grad_results = []

        for c in cs:
            for b in bs:
                b = np.array(b)
                res = grad_const(f, df, b, eps, n, c)
                const_grad_results.append((res, b, c))

        with open(root.joinpath("Data/grad_const.pickle"), "wb") as c:
            pickle.dump(const_grad_results, c)
    else:
        with open(root.joinpath("Data/grad_const.pickle"), 'rb') as c:
            const_grad_results = pickle.load(c)
    return const_grad_results


def grad_const_result_file():
    import ray
    @ray.remote
    def grad_const(f, df, x0, eps=1e-2, n=1000000, c=1e-5):
        """Gradient method with constant step"""
        xk = x0
        i = 0
        xis = []
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
            x_pred = xk
            sk = -df(xk)
            xk = xk + (c * sk)

        return xis, x0, c,

    if generate:
        ray.init()
        cs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  # step length
        bs = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [1, 7, -7], [0.1, 0.9, 1.1]]  # starting point
        result_ids = []
        for c in cs:
            for b in bs:
                b = np.array(b)
                result_ids.append(grad_const.remote(f, df, b, eps, n, c))

        result_ids = ray.get(result_ids)
        with open(root.joinpath("Data/grad_const_result.pickle"), "wb") as c:
            pickle.dump(result_ids, c)
    else:
        with open(root.joinpath("Data/grad_const_result.pickle"), 'rb') as c:
            result_ids = pickle.load(c)
    if generate:
        ray.shutdown()
    return result_ids

