class Metric:
    """Metric class to store the metric name and its value."""

    def __init__(self, name, value) -> None:
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name}: {self.value}"

    def __add__(self, other):
        if not isinstance(other, Metric):
            raise ValueError(f"Cannot add Metric with {type(other)}")

        return {self.name: self.value + other.value}

    def __div__(self, other):
        if not isinstance(other, Metric):
            raise ValueError(f"Cannot divide Metric with {type(other)}")

        return {self.name: self.value / other.value}

    def __rdiv__(self, other):
        if not isinstance(other, Metric):
            raise ValueError(f"Cannot divide Metric with {type(other)}")

        return {self.name: other / self.value}


if __name__ == "__main__":
    metric1 = Metric("OVRL", 0.5)
    metric2 = Metric("SIG", 0.6)

    print(metric1 + metric2)
    print(metric1 / metric2)
    print(metric1 + 1)
