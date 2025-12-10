from ..benchmark import Benchmark


from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import CircuitInstruction, Qubit

from qiskit.dagcircuit import DAGCircuit, DAGOpNode


def get_circuit_branch_probability(instruction, benchmark_name, backend_name) :
    return 0.5


def compute_circuit_object_depths(circuit : QuantumCircuit,
                                  benchmark_name : str,
                                  backend_name : str,
                                  count_ff: bool = False,
                                  ) -> float :

    if circuit is None : return {}, 0.0

    obj_depths = {obj: 0 for obj in [*circuit.qubits, *circuit.clbits]}
    total_depth = 0.0

    for instruction in circuit :
        if instruction.name == 'barrier' : continue

        objects = set([*instruction.qubits, *instruction.clbits])

        new_depth = max((obj_depths[obj] for obj in objects), default=0.0)

        if instruction.is_control_flow() and instruction.name == 'if_else' :
            if count_ff : new_depth += 1

            branch_probability = get_circuit_branch_probability(
                    instruction, benchmark_name, backend_name)

            _, if_subcircuit_total_depth = compute_circuit_object_depths(
                    instruction.params[0], benchmark_name, backend_name)
            _, else_subcircuit_total_depth = compute_circuit_object_depths(
                    instruction.params[1], benchmark_name, backend_name)

            added_depth = branch_probability * if_subcircuit_total_depth + \
                    (1-branch_probability) * else_subcircuit_total_depth

            new_depth += added_depth

        elif instruction.is_standard_gate() or \
            instruction.name in ['reset', 'measure', 'measure_2'] :
            new_depth += 1

        for obj in objects: obj_depths[obj] = new_depth
        total_depth = max(total_depth, new_depth)

    return obj_depths, total_depth


def compute_circuit_total_active_time(circuit : QuantumCircuit,
                                        benchmark_name : str,
                                        backend_name : str,
                                        count_ff: bool = False,
                                        ) -> float :

    if circuit is None : return 0.0

    qubit_activity = 0.0

    for instruction in circuit :
        if instruction.name in ['barrier', 'measure', 'measure_2'] : continue

        added_activity = 0.0
        if instruction.is_control_flow() and instruction.name == 'if_else' :
            if count_ff : added_activity += 1

            branch_probability = get_circuit_branch_probability(
                    instruction, benchmark_name, backend_name)

            if_subcircuit_total_activity = compute_circuit_total_active_time(
                    instruction.params[0], benchmark_name, backend_name)
            else_subcircuit_total_activity = compute_circuit_total_active_time(
                    instruction.params[1], benchmark_name, backend_name)

            added_activity += branch_probability * if_subcircuit_total_activity + \
                    (1 - branch_probability) * else_subcircuit_total_activity

        elif instruction.is_standard_gate() or instruction.name in ['reset'] :
            added_activity += instruction.operation.num_qubits

        qubit_activity += added_activity

    return qubit_activity


def compute_circuit_liveness(circuit : QuantumCircuit,
                             benchmark_name : str,
                             backend_name : str,
                             count_ff: bool = False,
                             ) -> float :
    circuit_qubit_depths, _ = compute_circuit_object_depths(
            circuit, benchmark_name, backend_name, count_ff=count_ff)
    total_gate_activity = compute_circuit_total_active_time(
            circuit, benchmark_name, backend_name, count_ff=count_ff)
    total_liveness = 0.0
    for obj, depth in circuit_qubit_depths.items() :
        if isinstance(obj, Qubit) :
            total_liveness += depth
    if total_liveness == 0.0 : return 0.0
    return total_gate_activity / total_liveness
