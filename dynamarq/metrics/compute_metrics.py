from ..benchmark import Benchmark


from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import CircuitInstruction

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
