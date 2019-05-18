class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb


    @staticmethod
    def prune(tree, mingain):
        # if branches aren't leaves, then prune them
        if tree.tb.results is None:
            decisionnode.prune(tree.tb, mingain)
        if tree.fb.results is None:
            decisionnode.prune(tree.fb, mingain)

        if tree.tb.results is not None and tree.fb.results is not None:
            tb, fb = [], []
            for v, c in tree.tb.results.items():
                tb += [[v]]*c
            for v, c in tree.fb.results.items():
                fb += [[v]]*c
        delta = loss_function()