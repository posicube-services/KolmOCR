class Metric:
    """
    Base class for metrics. Concrete metrics should implement run(args) and handle their own I/O.
    """

    name: str = "base"

    def run(self, args):
        raise NotImplementedError("Metric.run must be implemented.")
