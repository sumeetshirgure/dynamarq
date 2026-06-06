import math
import random
from dynamarq.benchmark import Benchmark
from dynamarq.clifford_dfe import clifford_dfe, expectation_from_counts

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.quantum_info import hellinger_fidelity, Operator

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

import guppylang
from guppylang import guppy

from guppylang.std.builtins import owned, array, result, comptime
from guppylang.std.quantum import qubit, measure_array, h, cx, s, sdg, x, z, toffoli

class RepeatUntilSuccess(Benchmark):
    """
    Implements repeat until success protocol from https://arxiv.org/pdf/1311.1074
    to implement a diagonal gate V=(I+2iZ)/sqrt(5) and its powers V^k.

    Device performance is measured using Hellinger fidelity to |0> state
    after a RUS implementation followed by a U3 gate reversing the effect.
    This is a small scale benchmark so we don't worry about scalability.
    """
    def __init__(self, reps: int) :
        self.n = 1
        self.a = 2
        self.k = reps # Implements V^k


    def name(self) :
        return f"RepeatUntilSuccess_{self.k}"


    def qiskit_circuits(self, mcm=True, stretch_dd=False) :
        """
        Get qiskit circuits to run on quantum hardware used in evaluating the benchmark score.
        """
        data = QuantumRegister(1, 'data')
        anc  = QuantumRegister(2, 'anc')
        c0   = ClassicalRegister(1, 'c0')
        c1   = ClassicalRegister(2, 'c1')
        qc   = QuantumCircuit(data, anc, c0, c1)

        V = [[(1 + 2j)/math.sqrt(5), 0], [0, (1 - 2j)/math.sqrt(5)]]
        W = Operator(V)

        qc.h(data)
        p = [ qc.add_stretch(f"p_{rep}") for rep in range(self.k) ]
        s = [ qc.add_stretch(f"s_{rep}") for rep in range(self.k) ]

        for rep in range(self.k) :
            qc.reset(anc)

            # Can't set classical bits in Qiskit so we flip ancillas before measurement.
            qc.x(anc)

            qc.barrier()

            if mcm :
                for i in range(2) :
                    qc.append(MidCircuitMeasure(), [anc[i]], [c1[i]])
            else :
                qc.measure(anc, c1)

            if stretch_dd :
                qc.delay(p[rep], data)
                qc.x(data)
                qc.delay(p[rep], data)
                qc.delay(p[rep], data)
                qc.x(data)
                qc.delay(p[rep], data)

            qc.barrier()

            with qc.while_loop(expr.not_equal(c1, 0)) :
                # Repeat until success.
                qc.reset(anc)
                qc.h(anc)
                qc.ccx(anc[0], anc[1], data)
                qc.s(data)
                qc.ccx(anc[0], anc[1], data)
                qc.z(data)
                qc.h(anc)

                qc.barrier()

                if mcm :
                    for i in range(2) :
                        qc.append(MidCircuitMeasure(), [anc[i]], [c1[i]])
                else :
                    qc.measure(anc, c1)

                if stretch_dd :
                    qc.delay(s[rep], data)
                    qc.x(data)
                    qc.delay(s[rep], data)
                    qc.delay(s[rep], data)
                    qc.x(data)
                    qc.delay(s[rep], data)
                qc.barrier()

        # Revert the operation for mirror circuit benchmarking.
        qc.unitary(W**(-self.k), data)
        qc.h(data)

        qc.measure(data, c0)

        return [qc]


    def qiskit_score(self, counts_list) :
        """Compute the Hellinger fidelity between the experimental and ideal
        qiskit results, i.e., 100% probabilty of measuring the zero state.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {"0": 1.0}
        fidelity_sum = 0.0
        for counts in counts_list :
            total_shots = sum(counts.values())
            device_hist = dict()
            for bitstr, count in counts.items() :
                data_qubits = bitstr[-1:-2:-1]
                if data_qubits not in device_hist :
                    device_hist[data_qubits] = 0
                device_hist[data_qubits] += count
            device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}
            fidelity_sum += hellinger_fidelity(ideal_dist, device_dist)
        return fidelity_sum / len(counts_list)


    def guppy_circuits(self) :
        return None


    def guppy_score(self, results_list) :
        return None
