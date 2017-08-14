class MyOptimizeResult(object):
    def __init__(self, res_gp):
        super(MyOptimizeResult, self).__init__()
        self.x_iters = res_gp.x_iters
        self.func_vals = res_gp.func_vals
        self.best_params = res_gp.x
