from dynamarq.benchmark import Benchmark

from math import pi

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit.quantum_info import hellinger_fidelity

from qiskit_aer import AerSimulator


class IPE(Benchmark):
    """
    Represents the iterative phase estimation benchmark for a toy unitary :
    (single qubit Z-rotation) with eigenstate |1> and eigenvalue exp(2*pi*i*theta/2**precision).
    Theta must be an integer between 0 and 2**precision - 1
    Device performance is based on the Hellinger fidelity between
    the experimental and ideal probability distributions.
    """
    def __init__(self, theta: float, precision: int) -> None:
        assert 1 <= precision <= 10, "Precision should be between 1 and 10."
        assert 0 <= theta < 2**precision, \
            "Theta should be an integer between 0 inclusive and 2**precision exclusive."

        self.theta = theta
        self.precision = precision


    def name(self) :
        return f"IPE_{self.theta}_{self.precision}"


    def qiskit_circuits(self, mcm=True, stretch_dd=False) :
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(self.precision)
        qc = QuantumCircuit(qreg, creg)
        qc.x(qreg[1])
        m = self.precision
        for k in range(1, m+1) :
            qc.h(qreg[0])
            theta_eff = 2*pi* (self.theta / 2**self.precision) * (2**(m-k))
            qc.cp(theta_eff, qreg[0], qreg[1])

            qc.barrier()

            for j in range(0, k) :
                with qc.if_test((creg[m-j-1], 1)) :
                    qc.p(-2*pi/2**(k-j), qreg[0])

            qc.h(qreg[0])

            if mcm :
                qc.append(MidCircuitMeasure(), [qreg[0]], [creg[m-k]])
            else :
                qc.measure(qreg[0], creg[m-k])

            if stretch_dd :
                for i in range(2) :
                    s = qc.add_stretch(f"s_{k}_{i}")
                    qc.delay(s, qreg[i])
                    qc.x(qreg[i])
                    qc.delay(s, qreg[i])
                    qc.delay(s, qreg[i])
                    qc.x(qreg[i])
                    qc.delay(s, qreg[i])

            qc.reset(qreg[0])

            qc.barrier()
        return [qc]


    def qiskit_score(self, counts_list) :
        r"""Compute the Hellinger fidelity between the experimental and ideal
        results, i.e., 50% probabilty of measuring the all-zero state and 50%
        probability of measuring the all-one state.

        The formula for the Hellinger fidelity between two distributions p and q
        is given by $(\sum_i{p_i q_i})^2$.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        counts = counts_list[0]
        fractional_part = self.theta
        ideal_dist = {f"{bin(fractional_part)[2:]:0>{self.precision}}"[::-1] : 1}
        total_shots = sum(counts.values())
        device_dist = {bitstr: count / total_shots for bitstr, count in counts.items()}
        return hellinger_fidelity(ideal_dist, device_dist)


    def guppy_circuits(self) :
        raise NotImplementedError("Iterative phase estimation is not available for guppy")


    def guppy_score(self, results) :
        raise NotImplementedError("Iterative phase estimation is not available for guppy")
