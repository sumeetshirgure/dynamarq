from dynamarq.benchmark import Benchmark

from math import pi

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

from qiskit_aer import AerSimulator

import guppylang
from guppylang import guppy

from guppylang.std.builtins import owned, array, result, comptime
from guppylang.std.quantum import qubit, measure, measure_array, h, x, z, cx, rx, rz, discard_array
from guppylang.std.angles import angle


class TFIM(Benchmark) :
    """Represents a quantum simulation benchmark for the 1D transverse
    field Ising model parameterized by the number of sites and the number
    of Trotter steps.
    
    Device performance is based on the relative error in the hardware
    computed average magnetization versus that from ideal circuit simulation
    based on matrix product states.

    The TFIM Hamiltonian reads: H = -J \\sum_i Z_i Z_{i+1} + h \\sum_i X_i
    It is Trotterized to the second order as e^(a+b) ~ e^(b/2) e^a e^(b/2)
    """
    def __init__(self, num_sites: int, num_steps: int) :
        self.n = num_sites
        self.steps = num_steps

        self.choices = [ (3, 2), (3, 5), (3, 20),
                         (5, 2), (5, 5), (5, 20),
                         (10, 2), (10, 5), (10, 20),
                         (30, 2), (30, 5), (30, 20) ]

        assert (self.n, self.steps) in self.choices, \
                f"(num_sites, num_steps) should be in {self.choices}"

        self.default_h = -7.0
        self.default_J = 1.0
        self.default_dt = 2 * pi * 1 / 30 * 0.25


    def name(self) :
        return f"TFIM_{self.n}_{self.steps}"


    def reference_circuits(self, h=None, J=None, dt=None) :
        if h is None : h = self.default_h
        if J is None : J = self.default_J
        if dt is None : dt = self.default_dt

        theta_x = h * dt
        theta_zz = -2 * J * dt

        data = QuantumRegister(self.n)
        meas = ClassicalRegister(self.n)
        circuit = QuantumCircuit(data, meas)

        for step in range(self.steps):
            for i in range(self.n):
                circuit.rx(theta_x, data[i])
            for i in range(0, self.n-1, 2) :
                circuit.rzz(theta_zz, data[i], data[i+1])
            for i in range(1, self.n-1, 2) :
                circuit.rzz(theta_zz, data[i], data[i+1])
            for i in range(self.n):
                circuit.rx(theta_x, data[i])

        circuit.measure(data, meas)

        return [circuit]


    def qiskit_circuits(self, h=None, J=None, dt=None, mcm=True, stretch_dd=False) :
        if h is None : h = self.default_h
        if J is None : J = self.default_J
        if dt is None : dt = self.default_dt

        theta_x = h * dt
        theta_zz = -2 * J * dt

        data = QuantumRegister(self.n, 'data')
        anc  = QuantumRegister(self.n-1, 'anc')
        meas = ClassicalRegister(self.n, 'meas')
        c1   = ClassicalRegister(self.n-1, 'c1')

        circuit = QuantumCircuit(data, anc, meas, c1)

        for step in range(self.steps) :

            for i in range(self.n) :
                circuit.rx(theta_x, data[i])
            for i in range(0, self.n-1, 2) :
                circuit.cx(data[i], anc[i])
                circuit.cx(data[i+1], anc[i])
                circuit.rz(theta_zz, anc[i])
                circuit.h(anc[i])
            for i in range(1, self.n-1, 2) :
                circuit.cx(data[i], anc[i])
                circuit.cx(data[i+1], anc[i])
                circuit.rz(theta_zz, anc[i])
                circuit.h(anc[i])

            circuit.barrier()

            for i in range(self.n-1) :
                if mcm :
                    circuit.append(MidCircuitMeasure(), [anc[i]], [c1[i]])
                else :
                    circuit.measure(anc[i], c1[i])

            if stretch_dd :
                for i in range(self.n) :
                    s = circuit.add_stretch(f"s_{step}_{i}")
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])
                    circuit.delay(s, data[i])
                    circuit.x(data[i])
                    circuit.delay(s, data[i])

            for i in range(0, self.n-1, 2) :
                flag = expr.lift(c1[i])
                with circuit.if_test(flag) :
                    circuit.z(data[i])
                    circuit.z(data[i+1])

            for i in range(1, self.n-1, 2) :
                flag = expr.lift(c1[i])
                with circuit.if_test(flag) :
                    circuit.z(data[i])
                    circuit.z(data[i+1])

            circuit.barrier()

            for i in range(self.n) :
                circuit.rx(theta_x, data[i])

        circuit.measure(data, meas)

        return [circuit]


    def guppy_circuits(self) :
        theta_x_over_pi = -0.11666666666666665
        theta_zz_over_pi = -0.03333333333333333

        n = guppy.nat_var('n')
        n1 = guppy.nat_var('n1')
        steps = guppy.nat_var('steps')

        @guppy
        def base_circuit(data: array[qubit, n] @owned,
                         anc: array[qubit, n1] @owned,
                         steps: int) -> array[qubit, n1] :

            for step in range(steps) :
                for i in range(n) :
                    rx(data[i], angle(-0.11666666666666665))
                for i in range(0, n1, 2) :
                    cx(data[i], anc[i])
                    cx(data[i+1], anc[i])
                    rz(anc[i], angle(-0.03333333333333333))
                    h(anc[i])
                for i in range(1, n1, 2) :
                    cx(data[i], anc[i])
                    cx(data[i+1], anc[i])
                    rz(anc[i], angle(-0.03333333333333333))
                    h(anc[i])
                cr = measure_array(anc)
                for i in range(0, n1, 2) :
                    if cr[i] :
                        z(data[i])
                        z(data[i+1])
                for i in range(1, n1, 2) :
                    if cr[i] :
                        z(data[i])
                        z(data[i+1])
                for i in range(n) :
                    rx(data[i], angle(-0.11666666666666665))
                anc = array(qubit() for _ in range(n1))

            meas = measure_array(data)
            for v in meas : result('meas', v)

            return anc

        @guppy.comptime
        def guppy_circuit() -> None :
            data = array(qubit() for _ in range(self.n))
            anc  = array(qubit() for _ in range(self.n-1))
            anc_final = base_circuit(data, anc, self.steps)
            discard_array(anc_final)

        return [guppy_circuit]


    def guppy_score(self, results_list) :
        results = results_list[0]

        collated_counts = results.collated_counts()
        total_shots = sum(collated_counts.values())

        device_hist = dict()
        for key in collated_counts.keys() :
            string = key[0][1]
            freq = collated_counts[ (('meas', string),) ]
            if string not in device_hist :
                device_hist[ string ] = 0
            device_hist[string] += freq

        mz_ideal = self.qiskit_average_magnetization(AerSimulator(method='matrix_product_state').run(
            self.reference_circuits()[0], shots=10000).result().get_counts())
        mz_exp = self.guppy_average_magnetization(device_hist)
        return 1 - abs(mz_ideal - mz_exp) / 2


    def guppy_average_magnetization(self, counts):
        mag = 0.0
        for index in range(self.n) :
            z_exp = 0.0
            tot = 0.0
            for bitstring, value in counts.items():
                bit = int(bitstring[index])
                sign = 1 if bit == 0 else -1
                z_exp += sign * value
                tot += value
            z_exp /= tot
            mag += z_exp
        return mag / self.n


    def qiskit_average_magnetization(self, counts):
        mag = 0.0
        for index in range(self.n) :
            z_exp = 0.0
            tot = 0.0
            for bitstring, value in counts.items():
                bit = int(bitstring[-1-index])
                sign = 1 if bit == 0 else -1
                z_exp += sign * value
                tot += value
            z_exp /= tot
            mag += z_exp
        return mag / self.n


    def qiskit_score(self, counts_list) :
        mz_ideal = self.qiskit_average_magnetization(AerSimulator(method='matrix_product_state').run(
                self.reference_circuits()[0], shots=10000).result().get_counts())
        mz_exp = self.qiskit_average_magnetization(counts_list[0])
        return 1 - abs(mz_ideal - mz_exp) / 2
