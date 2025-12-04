from dynamarq.benchmark import Benchmark

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit.quantum_info import hellinger_fidelity

import guppylang
from guppylang import guppy

from guppylang.std.builtins import owned, array, result, panic
from guppylang.std.quantum import qubit, measure, measure_array, h, cx, x


class GHZ(Benchmark) :
    """Represents the GHZ state preparation benchmark parameterized
    by the number of qubits n.

    Device performance is based on the Hellinger fidelity between
    the experimental and ideal probability distributions.
    """
    def __init__(self, num_qubits: int) :
        self.choices = [2, 3, 5, 10, 15, 20, 25, 30]
        self.n = num_qubits
        assert self.n in self.choices, f"n(={num_qubits}) must be among {self.choices}"


    def name(self) :
        return f"GHZ_{self.n}"


    def reference_circuits(self) :
        data = QuantumRegister(self.n, 'data')
        meas = ClassicalRegister(self.n, 'meas')
        circuit = QuantumCircuit(data, meas)

        circuit.h(data[0])
        for i in range(self.n-1) :
            circuit.cx(i, i+1)

        circuit.measure(data, meas)
        return [circuit]


    def qiskit_circuits(self, mcm = True, stretch_dd = False) :
        data = QuantumRegister(self.n, 'data')
        anc  = QuantumRegister(self.n-1, 'anc')
        meas = ClassicalRegister(self.n, 'meas')
        cr   = ClassicalRegister(self.n-1, 'cr')

        circuit = QuantumCircuit(data, anc, meas, cr)

        for i in range(self.n) :
            circuit.h(data[i])

        for i in range(self.n-1) :
            circuit.cx(data[i], anc[i])

        for i in range(self.n-1) :
            circuit.cx(data[i+1], anc[i])

        circuit.barrier()

        if mcm :
            for i in range(self.n-1) :
                circuit.append(MidCircuitMeasure(), [anc[i]], [cr[i]])
        else :
            circuit.measure(anc, cr)

        if stretch_dd :
            for i in range(self.n) :
                s = circuit.add_stretch(f"s_{i}")
                circuit.delay(s, data[i])
                circuit.x(data[i])
                circuit.delay(s, data[i])
                circuit.delay(s, data[i])
                circuit.x(data[i])
                circuit.delay(s, data[i])

        circuit.barrier()

        for i in range(self.n-1) :
            if i == 0 :
                parity = expr.lift(cr[i])
            else :
                parity = expr.bit_xor(cr[i], parity)
            with circuit.if_test(parity) :
                circuit.x(data[i+1])

        circuit.barrier()

        circuit.measure(data, meas)

        return [circuit]


    def guppy_circuits(self) :

        n = guppy.nat_var("n")
        n1 = guppy.nat_var("n1")

        @guppy
        def base_circuit(data: array[qubit, n] @owned, anc: array[qubit, n1] @owned) -> None :
            for i in range(n) : h(data[i])
            for i in range(n1) : cx(data[i], anc[i])
            for i in range(n1) : cx(data[i+1], anc[i])
            cr = measure_array(anc)
            parity = 0
            for i in range(n1) :
                parity = parity ^ int(cr[i])
                if parity == 1 : x(data[i+1])
            meas = measure_array(data)
            for v in meas : result('meas', v)

        @guppy
        def circuit_2() -> None :
            data = array(qubit() for _ in range(2))
            anc  = array(qubit() for _ in range(1))
            base_circuit(data, anc)

        @guppy
        def circuit_3() -> None :
            data = array(qubit() for _ in range(3))
            anc  = array(qubit() for _ in range(2))
            base_circuit(data, anc)

        @guppy
        def circuit_5() -> None :
            data = array(qubit() for _ in range(5))
            anc  = array(qubit() for _ in range(4))
            base_circuit(data, anc)

        @guppy
        def circuit_10() -> None :
            data = array(qubit() for _ in range(10))
            anc  = array(qubit() for _ in range(9))
            base_circuit(data, anc)

        @guppy
        def circuit_15() -> None :
            data = array(qubit() for _ in range(15))
            anc  = array(qubit() for _ in range(14))
            base_circuit(data, anc)

        @guppy
        def circuit_20() -> None :
            data = array(qubit() for _ in range(20))
            anc  = array(qubit() for _ in range(19))
            base_circuit(data, anc)

        @guppy
        def circuit_25() -> None :
            data = array(qubit() for _ in range(25))
            anc  = array(qubit() for _ in range(24))
            base_circuit(data, anc)

        @guppy
        def circuit_30() -> None :
            data = array(qubit() for _ in range(30))
            anc  = array(qubit() for _ in range(29))
            base_circuit(data, anc)

        match self.n :
            case 2  : return [circuit_2]
            case 3  : return [circuit_3]
            case 5  : return [circuit_5]
            case 10 : return [circuit_10]
            case 15 : return [circuit_15]
            case 20 : return [circuit_20]
            case 25 : return [circuit_25]
            case 30 : return [circuit_30]

        raise ValueError(f"Only choices are {self.choices}")

    def qiskit_score(self, counts_list) -> float:
        """Compute the Hellinger fidelity between the experimental and ideal
        qiskit results, i.e., 50% probabilty of measuring the all-zero state and 50%
        probability of measuring the all-one state.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {b * self.n: 0.5 for b in ["0", "1"]}

        fidelity_sum = 0.0

        for counts in counts_list :
            total_shots = sum(counts.values())

            device_hist = dict()

            for bitstr, count in counts.items() :
                data_qubits = bitstr[-1:-1-self.n:-1]
                if data_qubits not in device_hist :
                    device_hist[data_qubits] = 0
                device_hist[data_qubits] += count

            device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}

            fidelity_sum += hellinger_fidelity(ideal_dist, device_dist)

        return fidelity_sum / len(counts_list)


    def guppy_score(self, results_list) :
        """Compute the Hellinger fidelity between the experimental and ideal
        guppy results, i.e., 50% probabilty of measuring the all-zero state and 50%
        probability of measuring the all-one state.
        """
        # Create an equal weighted distribution between the all-0 and all-1 states
        ideal_dist = {b * self.n: 0.5 for b in ["0", "1"]}

        fidelity_sum = 0.0

        for results in results_list :
            collated_counts = results.collated_counts()

            total_shots = sum(collated_counts.values())

            device_hist = dict()

            for key in collated_counts.keys() :
                string = key[0][1]
                freq = collated_counts[ (('meas', string),) ]
                if string not in device_hist :
                    device_hist[ string ] = 0
                device_hist[string] += freq

            device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}

            fidelity_sum += hellinger_fidelity(ideal_dist, device_dist)

        return fidelity_sum / len(results_list)
