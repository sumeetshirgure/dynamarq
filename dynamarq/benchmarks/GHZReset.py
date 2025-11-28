from dynamarq.benchmark import Benchmark

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit.quantum_info import hellinger_fidelity

import guppylang
from guppylang import guppy

from guppylang.std.builtins import owned, array, result, panic
from guppylang.std.quantum import qubit, measure, measure_array, reset, h, cx, x


class GHZReset(Benchmark) :
    """Represents the GHZ state preparation with reset benchmark parameterized
    by the number of qubits n.

    Device performance is based on the Hellinger fidelity between
    the experimental and ideal probability distributions.
    """
    def __init__(self, num_qubits: int) :
        self.choices = [3, 5, 11, 15, 21, 25, 29]
        self.n = num_qubits
        assert self.n in self.choices, f"n(={num_qubits}) must be among {self.choices}"


    def reference_circuit(self) :
        data = QuantumRegister(self.n, 'data')
        meas = ClassicalRegister(self.n, 'meas')
        circuit = QuantumCircuit(data, meas)

        circuit.h(data[0])
        for i in range(self.n-1) :
            circuit.cx(i, i+1)

        circuit.measure(data, meas)
        return circuit


    def qiskit_circuits(self, mcm = True, stretch_dd = False) :
        assert self.n > 1 and self.n % 2 == 1, f"{self.n} must be an odd integer > 1"

        num_data_qubits = int( (self.n + 1) / 2 )
        num_ancillas    = self.n - num_data_qubits

        data = QuantumRegister(self.n, 'data')
        meas = ClassicalRegister(self.n, 'meas')
        cr   = ClassicalRegister(num_ancillas, 'cr')

        circuit = QuantumCircuit(data, meas, cr)

        for i in range(num_data_qubits) :
            circuit.h(data[2 * i])
        for i in range(num_data_qubits - 1):
            circuit.cx(data[i * 2], data[i * 2 + 1])
        for i in range(num_data_qubits - 1):
            circuit.cx(data[i * 2 + 2], data[i * 2 + 1])

        circuit.barrier()

        for i in range(num_ancillas):
            if mcm :
                circuit.append(MidCircuitMeasure(), [data[i * 2 + 1]], [cr[i]])
            else :
                circuit.measure(data[i * 2 + 1], cr[i])
            if i == 0:
                parity = expr.lift(cr[i])
            else:
                parity = expr.bit_xor(cr[i], parity)
            with circuit.if_test(parity):
                circuit.x(data[i * 2 + 2])

        if stretch_dd :
            for i in range(num_data_qubits) :
                s = circuit.add_stretch(f"s_{2*i}")
                circuit.delay(s, data[2*i])
                circuit.x(data[2*i])
                circuit.delay(s, data[2*i])
                circuit.delay(s, data[2*i])
                circuit.x(data[2*i])
                circuit.delay(s, data[2*i])

        for i in range(num_ancillas):
            circuit.reset(data[i * 2 + 1])
        for i in range(num_data_qubits - 1):
            circuit.cx(data[i * 2], data[i * 2 + 1])

        circuit.barrier()

        circuit.measure(data, meas)

        return [circuit]


    def guppy_circuits(self) :
        assert self.n > 1 and self.n % 2 == 1, f"{self.n} must be an odd integer > 1"

        @guppy
        def circuit_3() -> None :
            data = array(qubit() for _ in range(2))
            anc  = array(qubit() for _ in range(1))
            for i in range(1+3//2) : h(data[i])
            for i in range(3//2) : cx(data[i], anc[i])
            for i in range(3//2) : cx(data[i+1], anc[i])
            cr = measure_array(anc)
            parity = 0
            for i in range(3//2) :
                parity = parity ^ int(cr[i])
                if parity == 1 : x(data[i+1])
            anc  = array(qubit() for _ in range(1))
            for i in range(3//2) : cx(data[i], anc[i])
            meas0 = measure_array(data)
            meas1 = measure_array(anc)
            for i in range(3//2+1) :
                result('meas', meas0[i])
                if i < 3//2 :
                    result('meas', meas1[i])


        @guppy
        def circuit_5() -> None :
            data = array(qubit() for _ in range(3))
            anc  = array(qubit() for _ in range(2))
            for i in range(1+5//2) : h(data[i])
            for i in range(5//2) : cx(data[i], anc[i])
            for i in range(5//2) : cx(data[i+1], anc[i])
            cr = measure_array(anc)
            parity = 0
            for i in range(5//2) :
                parity = parity ^ int(cr[i])
                if parity == 1 : x(data[i+1])
            anc  = array(qubit() for _ in range(2))
            for i in range(5//2) : cx(data[i], anc[i])
            meas0 = measure_array(data)
            meas1 = measure_array(anc)
            for i in range(5//2+1) :
                result('meas', meas0[i])
                if i < 5//2 :
                    result('meas', meas1[i])


        @guppy
        def circuit_11() -> None :
            data = array(qubit() for _ in range(6))
            anc  = array(qubit() for _ in range(5))
            for i in range(1+11//2) : h(data[i])
            for i in range(11//2) : cx(data[i], anc[i])
            for i in range(11//2) : cx(data[i+1], anc[i])
            cr = measure_array(anc)
            parity = 0
            for i in range(11//2) :
                parity = parity ^ int(cr[i])
                if parity == 1 : x(data[i+1])
            anc  = array(qubit() for _ in range(5))
            for i in range(11//2) : cx(data[i], anc[i])
            meas0 = measure_array(data)
            meas1 = measure_array(anc)
            for i in range(11//2+1) :
                result('meas', meas0[i])
                if i < 11//2 :
                    result('meas', meas1[i])


        @guppy
        def circuit_15() -> None :
            data = array(qubit() for _ in range(8))
            anc  = array(qubit() for _ in range(7))
            for i in range(1+15//2) : h(data[i])
            for i in range(15//2) : cx(data[i], anc[i])
            for i in range(15//2) : cx(data[i+1], anc[i])
            cr = measure_array(anc)
            parity = 0
            for i in range(15//2) :
                parity = parity ^ int(cr[i])
                if parity == 1 : x(data[i+1])
            anc  = array(qubit() for _ in range(7))
            for i in range(15//2) : cx(data[i], anc[i])
            meas0 = measure_array(data)
            meas1 = measure_array(anc)
            for i in range(15//2+1) :
                result('meas', meas0[i])
                if i < 15//2 :
                    result('meas', meas1[i])


        @guppy
        def circuit_21() -> None :
            data = array(qubit() for _ in range(11))
            anc  = array(qubit() for _ in range(10))
            for i in range(1+21//2) : h(data[i])
            for i in range(21//2) : cx(data[i], anc[i])
            for i in range(21//2) : cx(data[i+1], anc[i])
            cr = measure_array(anc)
            parity = 0
            for i in range(21//2) :
                parity = parity ^ int(cr[i])
                if parity == 1 : x(data[i+1])
            anc  = array(qubit() for _ in range(10))
            for i in range(21//2) : cx(data[i], anc[i])
            meas0 = measure_array(data)
            meas1 = measure_array(anc)
            for i in range(21//2+1) :
                result('meas', meas0[i])
                if i < 21//2 :
                    result('meas', meas1[i])


        @guppy
        def circuit_25() -> None :
            data = array(qubit() for _ in range(13))
            anc  = array(qubit() for _ in range(12))
            for i in range(1+25//2) : h(data[i])
            for i in range(25//2) : cx(data[i], anc[i])
            for i in range(25//2) : cx(data[i+1], anc[i])
            cr = measure_array(anc)
            parity = 0
            for i in range(25//2) :
                parity = parity ^ int(cr[i])
                if parity == 1 : x(data[i+1])
            anc  = array(qubit() for _ in range(12))
            for i in range(25//2) : cx(data[i], anc[i])
            meas0 = measure_array(data)
            meas1 = measure_array(anc)
            for i in range(25//2+1) :
                result('meas', meas0[i])
                if i < 25//2 :
                    result('meas', meas1[i])


        @guppy
        def circuit_29() -> None :
            data = array(qubit() for _ in range(15))
            anc  = array(qubit() for _ in range(14))
            for i in range(1+29//2) : h(data[i])
            for i in range(29//2) : cx(data[i], anc[i])
            for i in range(29//2) : cx(data[i+1], anc[i])
            cr = measure_array(anc)
            parity = 0
            for i in range(29//2) :
                parity = parity ^ int(cr[i])
                if parity == 1 : x(data[i+1])
            anc  = array(qubit() for _ in range(14))
            for i in range(29//2) : cx(data[i], anc[i])
            meas0 = measure_array(data)
            meas1 = measure_array(anc)
            for i in range(29//2+1) :
                result('meas', meas0[i])
                if i < 29//2 :
                    result('meas', meas1[i])

        self.choices = [3, 5, 11, 15, 21, 25, 29]

        match self.n :
            case 3  : return [circuit_3]
            case 5  : return [circuit_5]
            case 11 : return [circuit_11]
            case 15 : return [circuit_15]
            case 21 : return [circuit_21]
            case 25 : return [circuit_25]
            case 29 : return [circuit_29]

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
