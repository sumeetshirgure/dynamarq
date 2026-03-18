from dynamarq.benchmark import Benchmark

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

import guppylang
from guppylang import guppy

from guppylang.std.builtins import owned, array, result, comptime
from guppylang.std.quantum import qubit, measure, measure_array, h, cx, cz, x, y, z


class SteaneCode(Benchmark) :
    """Represents the [[7,1,3]] Steane code.
    This benchmark evaluates how well the hardware preserves |1> and |+> states
    encoded in the Steane code.
    Performs one round of syndrome measurement and logical operator measurement.

    We evaluate the logical error rate as the score for this benchmark.
    """
    def __init__(self) :
        self.init_states = ['1', '+']

    def name(self) :
        return f'SteaneCode'

    def qiskit_circuits(self, mcm=True, stretch_dd=False) :

        circuits = []
        for init_state in self.init_states :
            data  = QuantumRegister(7, 'data')
            xanc  = QuantumRegister(3, 'xanc')
            zanc  = QuantumRegister(3, 'zanc')
            op    = QuantumRegister(1, 'log')
            result = ClassicalRegister(1, 'result')
            meas   = ClassicalRegister(7, 'meas')
            xsyn    = ClassicalRegister(3, 'xsyn')
            zsyn    = ClassicalRegister(3, 'zsyn')

            circuit = QuantumCircuit(data, xanc, zanc, op, result, meas, xsyn, zsyn)

            circuit.barrier()

            # Prepare logical |0>
            circuit.h(data[[0,1,3]])
            circuit.cx(data[0], data[2])
            circuit.cx(data[0], data[4])
            circuit.cx(data[0], data[6])
            circuit.cx(data[1], data[2])
            circuit.cx(data[1], data[5])
            circuit.cx(data[1], data[6])
            circuit.cx(data[3], data[4])
            circuit.cx(data[3], data[5])
            circuit.cx(data[3], data[6])

            if init_state == '1' :
                circuit.x(data) # Logical X
            if init_state == '+' :
                circuit.h(data) # Logical H

            circuit.barrier()
            
            checks = [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]]

            # Z-Syndromes (Detect X errors)
            for i, targets in enumerate(checks):
                for q in targets:
                    circuit.cx(data[q], xanc[i])
                
            # X-Syndromes (Detect Z errors)
            for i, targets in enumerate(checks):
                circuit.h(zanc[i])
                for q in targets:
                    circuit.cx(zanc[i], data[q])
                circuit.h(zanc[i])

            if mcm :
                for i in range(3) :
                    circuit.append(MidCircuitMeasure(), [xanc[i]], [xsyn[i]])
                for i in range(3) :
                    circuit.append(MidCircuitMeasure(), [zanc[i]], [zsyn[i]])
            else :
                circuit.measure(xanc, xsyn)
                circuit.measure(zanc, zsyn)

            if stretch_dd :
                for i in range(7) :
                    s = circuit.add_stretch(f"s_syn_{i}")
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])

            circuit.barrier()

            # Map binary syndrome (1-7) to qubit index (0-6)
            # Note: syn[0:3] is bit-flip, syn[3:6] is phase-flip
            syndrome_map = {1: 3, 2: 1, 3: 5, 4: 0, 5: 4, 6: 2, 7: 6}

            for syn_val, qubit_idx in syndrome_map.items():
                # Correct Bit-flips (X)
                with circuit.if_test((xsyn, syn_val)):
                    circuit.x(data[qubit_idx])
                # Correct Phase-flips (Z)
                with circuit.if_test((zsyn, syn_val)):
                    circuit.z(data[qubit_idx])

            circuit.barrier()

            # Revert logical state to |0>
            if init_state == '1' :
                circuit.x(data) # Logical X
            if init_state == '+' :
                circuit.h(data) # Logical H

            circuit.h(op)
            for i in range(7) :
                circuit.cz(data[i], op)
            circuit.h(op)
            circuit.measure(op, result)
            circuit.measure(data, meas)

            circuits.append(circuit)

        return circuits


    def qiskit_score(self, counts_list) :

        fidelity_sum = 0.0
        for init_state, counts in zip(self.init_states, counts_list) :
            total_shots = sum(counts.values())

            device_hist = dict()
            for bitstring, count in counts.items() :
                bit = bitstring[-1]
                if bit not in device_hist :
                    device_hist[bit] = 0
                device_hist[bit] += count
            device_dist = {bit: count / total_shots for bit, count in device_hist.items()}

            fidelity_sum += device_dist['0']

        return fidelity_sum / len(counts_list)


    def guppy_circuits(self) :

        @guppy
        def steane_code() -> None :
            data = array(qubit() for _ in range(7))
            xanc = array(qubit() for _ in range(3))
            zanc = array(qubit() for _ in range(3))
            op = qubit()

            # Prepare logical |0>
            h(data[0])
            h(data[1])
            h(data[3])
            cx(data[0], data[2])
            cx(data[0], data[4])
            cx(data[0], data[6])
            cx(data[1], data[2])
            cx(data[1], data[5])
            cx(data[1], data[6])
            cx(data[3], data[4])
            cx(data[3], data[5])
            cx(data[3], data[6])

            if comptime(init_state == '1') :
                for i in range(7) :
                    x(data[i]) # Logical X
            if comptime(init_state == '+') :
                for i in range(7) :
                    h(data[i]) # Logical H

            # Syndrome checks and corrections
            checks = ((3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6))

            # Z-Syndromes (Detect X errors)
            cx(data[3], xanc[0])
            cx(data[4], xanc[0])
            cx(data[5], xanc[0])
            cx(data[6], xanc[0])
            cx(data[1], xanc[1])
            cx(data[2], xanc[1])
            cx(data[5], xanc[1])
            cx(data[6], xanc[1])
            cx(data[0], xanc[2])
            cx(data[2], xanc[2])
            cx(data[4], xanc[2])
            cx(data[6], xanc[2])

            h(zanc[0])
            h(zanc[1])
            h(zanc[2])
            cx(zanc[0], data[3])
            cx(zanc[0], data[4])
            cx(zanc[0], data[5])
            cx(zanc[0], data[6])
            cx(zanc[1], data[1])
            cx(zanc[1], data[2])
            cx(zanc[1], data[5])
            cx(zanc[1], data[6])
            cx(zanc[2], data[0])
            cx(zanc[2], data[2])
            cx(zanc[2], data[4])
            cx(zanc[2], data[6])
            h(zanc[0])
            h(zanc[1])
            h(zanc[2])

            xsyn = measure_array(xanc)
            zsyn = measure_array(zanc)
            x_syn = int(xsyn[0]) + int(xsyn[1])<<1 + int(xsyn[2])<<2
            z_syn = int(zsyn[0]) + int(zsyn[1])<<1 + int(zsyn[2])<<2

            # Map binary syndrome (1-7) to qubit index (0-6)
            # syndrome_map = {1: 3, 2: 1, 3: 5, 4: 0, 5: 4, 6: 2, 7: 6}

            if x_syn == 1 : x(data[3])
            if z_syn == 1 : z(data[3])
            if x_syn == 2 : x(data[1])
            if z_syn == 2 : z(data[1])
            if x_syn == 3 : x(data[5])
            if z_syn == 3 : z(data[5])
            if x_syn == 4 : x(data[0])
            if z_syn == 4 : z(data[0])
            if x_syn == 5 : x(data[4])
            if z_syn == 5 : z(data[4])
            if x_syn == 6 : x(data[2])
            if z_syn == 6 : z(data[2])
            if x_syn == 7 : x(data[6])
            if z_syn == 7 : z(data[6])

            if comptime(init_state == '1') :
                for i in range(7) :
                    x(data[i]) # Logical X
            if comptime(init_state == '+') :
                for i in range(7) :
                    h(data[i]) # Logical H

            h(op)
            for i in range(7) :
                cz(data[i], op)
            h(op)

            res = measure(op)
            result('result', res)
            meas = measure_array(data)

        init_state = '1'
        @guppy
        def sc1() -> None :
            steane_code() 
        sc1 = sc1.compile()

        init_state = '+'
        @guppy
        def scp() -> None :
            steane_code() 
        scp = scp.compile()

        return [sc1, scp]


    def guppy_score(self, results_list) :

        fidelity_sum = 0.0

        for init_state, results in zip(self.init_states, results_list) :
            collated_counts = results.collated_counts()
            total_shots = sum(collated_counts.values())

            device_hist = dict()

            for key in collated_counts.keys() :
                string = key[0][1]
                freq = collated_counts[ (('result', string),) ]
                if string not in device_hist :
                    device_hist[ string ] = 0
                device_hist[ string ] += freq

            device_dist = {bitstring: count/total_shots for bitstring, count in device_hist.items()}

            fidelity_sum += device_dist['0']

        return fidelity_sum / len(results_list)
