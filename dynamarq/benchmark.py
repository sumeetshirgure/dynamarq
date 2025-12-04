import abc

class Benchmark:
    """Class representing a quantum benchmark application.

    Concrete subclasses must implement the abstract methods
    ``reference_circuit()``, ``qiskit_circuit()``, ``guppy_circuit()``, and ``score()``.
    """
    @abc.abstractmethod
    def name(self) :
        """Returns the name of the benchmark including its parameters."""

    @abc.abstractmethod
    def reference_circuits(self) :
        """Returns a simplified circuit with identical measurement statistics used for reference."""

    @abc.abstractmethod
    def qiskit_circuits(self) :
        """Returns a sequence of Qiskit circuits with the current benchmark parameters."""

    @abc.abstractmethod
    def guppy_circuits(self) :
        """Returns a sequence of guppy circuits with the current benchmark parameters."""

    @abc.abstractmethod
    def qiskit_score(self, counts) :
        """Returns a normalized [0,1] score reflecting device performance.

        Args:
            counts: Sequence of dictionaries containing the measurement counts from qiskit execution.
        """

    @abc.abstractmethod
    def guppy_score(self, counts) :
        """Returns a normalized [0,1] score reflecting device performance.

        Args:
            counts: Sequence of dictionaries containing the measurement counts from guppy execution.
        """
