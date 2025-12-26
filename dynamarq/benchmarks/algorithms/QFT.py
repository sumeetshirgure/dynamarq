from dynamarq.benchmark import Benchmark

import random
from math import pi

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit.quantum_info import hellinger_fidelity


class QFT(Benchmark):
    """Represents the dynamic Quantum Fourier transform benchmark parameterized
    by the number of qubits n, and number of states num_states.

    Device performance is based on the Hellinger fidelity between
    the experimental and ideal probability distributions.
    """

    def __init__(self, n: int, num_states: int = 3) :
        self.n = n
        self.secret_ints = []

        random.seed(42)
        for i in range (num_states):
            self.secret_ints.append(random.randint(1, 2**self.n - 1))


    def name(self) :
        return f"QFT_{self.n}"


    def qiskit_circuits(self, mcm=True, stretch_dd=False) :
        circuits = []
        for s in self.secret_ints:
            qr = QuantumRegister(self.n)
            cr = ClassicalRegister(self.n, name='meas')
            circuit = QuantumCircuit(qr, cr)

            for i in range(self.n):
                circuit.h(i)

            for i_q in range(0, self.n):
                divisor = 2 ** (i_q)
                circuit.rz(s * pi / divisor, qr[i_q])

            dynamic_inv_qft = self._dyn_inv_qft_gate(mcm, stretch_dd)
            circuit.compose(dynamic_inv_qft, qubits=qr, clbits=cr, inplace=True)

            circuits.append(circuit)
        return circuits


    def _dyn_inv_qft_gate(self, mcm=True, stretch_dd=False):
        qr = QuantumRegister(self.n, name="q_dyn_inv")
        cr = ClassicalRegister(self.n, name='meas')
        qc = QuantumCircuit(qr, cr, name="dyn_inv_qft")

        # mirror the static inv-QFT loop order, but with mid-circuit feed-forward
        for i_qubit in reversed(range(self.n)):
            hidx = self.n - 1 - i_qubit

            qc.h(qr[hidx])

            qc.barrier()

            if mcm :
                qc.append(MidCircuitMeasure(), [qr[hidx]], [cr[hidx]])
            else :
                qc.measure(qr[hidx], cr[hidx])

            if stretch_dd :
                for i in range(self.n) :
                    s = qc.add_stretch(f"s_{hidx}_{i}")
                    qc.delay(s, qr[i])
                    qc.x(qr[i])
                    qc.delay(s, qr[i])
                    qc.delay(s, qr[i])
                    qc.x(qr[i])
                    qc.delay(s, qr[i])

            if hidx < self.n - 1:
                for j in reversed(range(i_qubit)):
                    theta = pi / (2 ** (i_qubit - j))
                    with qc.if_test((cr[hidx], 1)):
                        qc.rz(-theta, qr[self.n - 1 - j])

            qc.barrier()
        return qc


    def qiskit_score(self, counts_list) :
        """Compute the Hellinger fidelity between the experimental and ideal results.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        hfs = []
        for s, counts in zip(self.secret_ints, counts_list):
            key = format(s, f"0{self.n}b")
            # correct distribution is measuring the key 100% of the time
            correct_dist = {key: 1.0}
            hfs.append(hellinger_fidelity(counts, correct_dist))

        return sum(hfs) / len(hfs)


    def guppy_circuits(self) :
        raise NotImplementedError("QFT benchmark is not available for guppy")


    def guppy_score(self) :
        raise NotImplementedError("QFT benchmark is not available for guppy")
