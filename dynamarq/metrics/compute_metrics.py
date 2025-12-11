from ..benchmark import Benchmark

import networkx as nx

from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import CircuitInstruction, Qubit

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit


def get_instruction_branch_probability(instruction, benchmark_name, backend_name) :
    assert backend_name in ['ibm_kingston', 'ibm_pittsburgh']

    # These values are obtained from IBM Quantum platform
    if backend_name == 'ibm_kingston'   : p, m = 2.01e-3, 8.911e-3
    if backend_name == 'ibm_pittsburgh' : p, m = 1.61e-3, 3.662e-3

    if benchmark_name == 'RepetitionCode_3' :
        bp = 2 * p * (1-p)**3 * (1-m) ** 2 + (1-p)**4 * m * (1-m)
        return bp

    if benchmark_name == 'RepetitionCode_5' :
        bp = 2 * p * (1-p)**7 * (1-m) ** 4 + (1-p) ** 8 * m * (1-m)**3
        return bp

    if benchmark_name == 'FiveQubitCode' :
        if instruction.clbits[0]._register.name == 'syn' :
            return 1.0 / 16.0

    return 0.5


def get_node_branch_probability(node, benchmark_name, backend_name) :
    assert backend_name in ['ibm_kingston', 'ibm_pittsburgh']

    # These values are obtained from IBM Quantum platform
    if backend_name == 'ibm_kingston'   : p, m = 2.01e-3, 8.911e-3
    if backend_name == 'ibm_pittsburgh' : p, m = 1.61e-3, 3.662e-3

    if benchmark_name == 'RepetitionCode_3' :
        bp = 2 * p * (1-p)**3 * (1-m) ** 2 + (1-p)**4 * m * (1-m)
        return bp

    if benchmark_name == 'RepetitionCode_5' :
        bp = 2 * p * (1-p)**7 * (1-m) ** 4 + (1-p) ** 8 * m * (1-m)**3
        return bp

    if benchmark_name == 'FiveQubitCode' :
        if node.cargs[0]._register.name == 'syn' :
            return 1.0 / 16.0

    return 0.5


def compute_circuit_object_depths(circuit : QuantumCircuit,
                                  benchmark_name : str,
                                  backend_name : str,
                                  count_ff: bool = False,
                                  ) -> (dict, float) :

    if circuit is None : return {}, 0.0

    obj_depths = {obj: 0 for obj in [*circuit.qubits, *circuit.clbits]}
    total_depth = 0.0

    for instruction in circuit :
        if instruction.name == 'barrier' : continue

        objects = set([*instruction.qubits, *instruction.clbits])

        new_depth = max((obj_depths[obj] for obj in objects), default=0.0)

        if instruction.is_control_flow() and instruction.name == 'if_else' :
            if count_ff : new_depth += 1

            branch_probability = get_instruction_branch_probability(
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

            branch_probability = get_instruction_branch_probability(
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


def compute_circuit_num_gates(circuit : QuantumCircuit,
                              benchmark_name : str,
                              backend_name : str,
                              count_measure: bool = True,
                              count_reset: bool = True,
                              count_ff : bool = False,
                              ) -> float :

    if circuit is None : return 0.0

    num_gates = 0.0

    for instruction in circuit :
        if instruction.name in ['barrier'] : continue

        if instruction.name in ['measure', 'measure_2'] and count_measure :
            num_gates += 1.0
            continue

        if instruction.name in ['reset'] and count_reset :
            num_gates += 1.0
            continue

        added_num_gates = 0.0
        if instruction.is_control_flow() and instruction.name == 'if_else' :
            if count_ff : added_num_gates += 1

            branch_probability = get_instruction_branch_probability(
                    instruction, benchmark_name, backend_name)

            if_subcircuit_total_gates = compute_circuit_num_gates(
                    instruction.params[0], benchmark_name, backend_name,
                    count_measure, count_reset, count_ff)
            else_subcircuit_total_gates = compute_circuit_num_gates(
                    instruction.params[1], benchmark_name, backend_name,
                    count_measure, count_reset, count_ff)

            added_num_gates += branch_probability * if_subcircuit_total_gates + \
                    (1 - branch_probability) * else_subcircuit_total_gates

        elif instruction.is_standard_gate() :
            added_num_gates += 1.0

        num_gates += added_num_gates

    return num_gates


def get_circuit_critical_path(circuit : QuantumCircuit,
                              benchmark_name : str,
                              backend_name : str,
                              count_ff : bool = False,
                              ) -> (list, float) :

    if circuit is None :
        return [], 0.0

    dag = circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")
    topo_nodes = list(dag.topological_op_nodes())

    longest_distance = {node: 0 for node in topo_nodes}
    predecessor = {node: None for node in topo_nodes}

    for node in topo_nodes :
        for succ in dag.successors(node) :
            if not isinstance(succ, DAGOpNode) : continue

            new_distance = longest_distance[node]
            if succ.is_control_flow() and succ.name == 'if_else' :
                if count_ff : new_distance += 1

                branch_probability = get_node_branch_probability(
                        succ, benchmark_name, backend_name)

                _, if_subcircuit_critical_depth = get_circuit_critical_path(
                        succ.params[0], benchmark_name, backend_name, count_ff)
                _, else_subcircuit_critical_depth = get_circuit_critical_path(
                        succ.params[1], benchmark_name, backend_name, count_ff)

                added_distance = branch_probability * if_subcircuit_critical_depth + \
                        (1 - branch_probability) * else_subcircuit_critical_depth

                new_distance += added_distance
            
            elif succ.is_standard_gate() or succ.name in ['measure', 'measure_2', 'reset'] :
                new_distance += 1

            if new_distance > longest_distance[succ]:
                longest_distance[succ] = new_distance
                predecessor[succ] = node

    if not longest_distance :
        return [], 0

    end_node = max(longest_distance, key=longest_distance.get)
    critical_depth = longest_distance[end_node]

    critical_path = []
    while end_node is not None:
        critical_path.append(end_node)
        end_node = predecessor[end_node]
    critical_path.reverse()

    return critical_path, critical_depth



def compute_classical_entanglement_gates(circuit : QuantumCircuit,
                                         benchmark_name : str,
                                         backend_name : str,
                                         count_measure: bool = True,
                                         count_reset: bool = True,
                                         count_ff: bool = False,
                                         ) -> float:
    num_branch_gates = 0.0

    for instruction in circuit :
        if instruction.is_control_flow() and instruction.name == 'if_else' :

            branch_probability = get_instruction_branch_probability(
                    instruction, benchmark_name, backend_name)

            if_subcircuit_total_gates = compute_circuit_num_gates(
                    instruction.params[0], benchmark_name, backend_name,
                    count_measure, count_reset, count_ff)
            else_subcircuit_total_gates = compute_circuit_num_gates(
                    instruction.params[1], benchmark_name, backend_name,
                    count_measure, count_reset, count_ff)

            num_branch_gates += branch_probability * if_subcircuit_total_gates + \
                    (1 - branch_probability) * else_subcircuit_total_gates 

    return num_branch_gates


def compute_circuit_critical_depth_quantum(circuit : QuantumCircuit,
                                           benchmark_name : str,
                                           backend_name : str,
                                           ) -> float :
    critical_path, _ = get_circuit_critical_path(
            circuit, benchmark_name, backend_name, count_ff=False)

    num_two_qubit_longest_path = 0
    for node in critical_path :
        if node.op.num_qubits > 1 :
            num_two_qubit_longest_path += 1

    num_two_qubits_total = 0
    for instruction in circuit._data:
        if instruction.operation.num_qubits > 1:
            num_two_qubits_total += 1

    if num_two_qubits_total == 0:
        return 0
    return num_two_qubit_longest_path / num_two_qubits_total


def compute_circuit_critical_depth_quantum_classical(circuit : QuantumCircuit,
                                                     benchmark_name : str,
                                                     backend_name : str,
                                                     count_measure : bool = True,
                                                     count_reset : bool = True,
                                                     count_ff : bool = False
                                                     ) -> float :
    critical_path, _ = get_circuit_critical_path(
            circuit, benchmark_name, backend_name, count_ff=count_ff)

    num_two_qubit_longest_path = 0
    for node in critical_path :

        if node.op.num_qubits > 1:
            num_two_qubit_longest_path += 1

        elif node.op.name == 'if_else' :
            branch_probability = get_node_branch_probability(
                    node, benchmark_name, backend_name)

            if_subcircuit_total_gates = compute_circuit_num_gates(
                    node.params[0], benchmark_name, backend_name,
                    count_measure, count_reset, count_ff)
            else_subcircuit_total_gates = compute_circuit_num_gates(
                    node.params[1], benchmark_name, backend_name,
                    count_measure, count_reset, count_ff)

            num_two_qubit_longest_path += branch_probability * if_subcircuit_total_gates + \
                    (1 - branch_probability) * else_subcircuit_total_gates

    num_two_qubits_total = 0
    for instruction in circuit :
        if instruction.operation.num_qubits > 1:
            num_two_qubits_total += 1

    num_two_qubits_total += compute_classical_entanglement_gates(
            circuit, benchmark_name, backend_name,
            count_measure, count_reset, count_ff)

    if num_two_qubits_total == 0:
        return 0
    return num_two_qubit_longest_path / num_two_qubits_total


def compute_circuit_mcm_depth_ratio(circuit : QuantumCircuit,
                                    benchmark_name : str,
                                    backend_name : str,
                                    count_ff:bool = False,
                                    ) -> float :
    dag = circuit_to_dag(circuit)
    mid_measurement_depth = 0
    for layer in dag.layers():
        layer_ops = layer['graph'].op_nodes()
        for node in layer_ops:
            if node.name == 'measure_2' :
                mid_measurement_depth += 1
                break
    _, circuit_depth = compute_circuit_object_depths(
            circuit, benchmark_name, backend_name, count_ff=count_ff)
    if circuit_depth == 0 :
        return 0
    return mid_measurement_depth / circuit_depth


def compute_circuit_mcm_plus_ff_depth_ratio(circuit : QuantumCircuit,
                                            benchmark_name : str,
                                            backend_name : str
                                            ) -> float :
    dag = circuit_to_dag(circuit)
    mcm_ff_depth = 0
    for layer in dag.layers():
        layer_ops = layer['graph'].op_nodes()
        for node in layer_ops:
            if node.name == 'measure_2' :
                mcm_ff_depth += 1
                break
            elif node.is_control_flow() and node.op.name == 'if_else' :
                mcm_ff_depth += 1
                break
    _, circuit_depth = compute_circuit_object_depths(
            circuit, benchmark_name, backend_name, count_ff=True)
    if circuit_depth == 0 :
        return 0
    return mcm_ff_depth / circuit_depth


def compute_circuit_parallelism(circuit : QuantumCircuit,
                                benchmark_name : str,
                                backend_name : str,
                                count_ff: bool = False
                                ) -> float:
    _, depth = compute_circuit_object_depths(
            circuit, benchmark_name, backend_name, count_ff=count_ff)
    num_gates = compute_circuit_num_gates(
            circuit, benchmark_name, backend_name, count_ff=count_ff)
    if circuit.num_qubits <= 1 : return 0
    return max((num_gates / depth - 1) / (circuit.num_qubits - 1), 0)


def get_connectivity_graph(circuit : QuantumCircuit,
                           benchmark_name : str,
                           backend_name : str,
                           ) -> nx.Graph :
    dag = circuit_to_dag(circuit)
    dag.remove_all_ops_named('barrier')
    graph = nx.Graph()
    clbit_map = dict()
    graph.add_nodes_from(dag.qubits)
    for node in dag.topological_op_nodes() :
        if node.op.name in ['measure', 'measure_2'] :
            clbit_map[node.cargs[0]] = node.qargs[0]
        elif node.is_control_flow() and node.op.name == 'if_else' :
            branch_probability = get_node_branch_probability(
                    node, benchmark_name, backend_name)
            for q0 in node.qargs :
                for clbit in node.cargs :
                    if clbit not in clbit_map : continue
                    q1 = clbit_map[clbit]
                    edge = (q0, q1)
                    if edge not in graph.edges() :
                        graph.add_edge(edge[0], edge[1], weight=branch_probability)
        elif node.is_standard_gate() :
            if node.op.num_qubits == 2 :
                q0, q1 = node.qargs
                edge = (q0, q1)
                if edge not in graph.edges() :
                    graph.add_edge(edge[0], edge[1], weight=1.0)
    return graph


def compute_circuit_communication(circuit : QuantumCircuit,
                                  benchmark_name : str,
                                  backend_name : str,
                                  ) -> float:
    num_qubits = circuit.num_qubits
    graph = get_connectivity_graph(circuit, benchmark_name, backend_name)
    degree_sum = sum([graph.degree(n, weight='weight') for n in graph.nodes()])
    if num_qubits <= 1 :
        return 0
    return degree_sum / (num_qubits * (num_qubits - 1))


def compute_circuit_quantum_entanglement(circuit : QuantumCircuit,
                                         benchmark_name : str,
                                         backend_name : str,
                                         count_measure : bool = True,
                                         count_reset : bool = True,
                                         count_ff : bool = False
                                         ) -> float :
    num_two_qubit_gates = 0
    num_gates = compute_circuit_num_gates(
            circuit, benchmark_name, backend_name,
            count_measure, count_reset, count_ff)
    for instruction in circuit :
        if instruction.is_standard_gate() and instruction.operation.num_qubits == 2 :
            num_two_qubit_gates += 1
    if num_gates == 0 : return 0
    return num_two_qubit_gates / num_gates


def compute_circuit_quantum_classical_entanglement(circuit : QuantumCircuit,
                                                   benchmark_name : str,
                                                   backend_name : str,
                                                   count_measure : bool = True,
                                                   count_reset : bool = True,
                                                   count_ff : bool = False
                                                   ) -> float :
    quantum_entanglement = compute_circuit_quantum_entanglement(
            circuit, benchmark_name, backend_name,
            count_measure, count_reset, count_ff)
    num_gates = compute_circuit_num_gates(
            circuit, benchmark_name, backend_name,
            count_measure, count_reset, count_ff)
    classical_entanglement_gates = compute_classical_entanglement_gates(
            circuit, benchmark_name, backend_name,
            count_measure, count_reset, count_ff)
    return quantum_entanglement + classical_entanglement_gates / num_gates


def get_metric_names() :
    return [
            'depth',
            'depth_ff',
            'liveness',
            'liveness_ff',
            'num_gates',
            'num_gates_measure_reset',
            'num_gates_measure_reset_ff',
            'critical_path_quantum',
            'critical_path_quantum_classical',
            'mcm_depth_ratio',
            'mcm_depth_ratio_ff',
            'mcm_plus_ff_depth_ratio',
            'parallelism',
            'parallelism_ff',
            'communication',
            'quantum_entanglement',
            'quantum_entanglement_measure_reset',
            'quantum_entanglement_measure_reset_ff',
            'quantum_classical_entanglement',
            'quantum_classical_entanglement_measure_reset',
            'quantum_classical_entanglement_measure_reset_ff',
            ]


def get_circuit_metrics(circuit : QuantumCircuit,
                        benchmark_name : str,
                        backend_name : str
                        ) -> dict :
    metrics = dict()

    metrics['depth'] = compute_circuit_object_depths(
            circuit, benchmark_name, backend_name, count_ff=False)[1]

    metrics['depth_ff'] = compute_circuit_object_depths(
            circuit, benchmark_name, backend_name, count_ff=True)[1]

    metrics['liveness'] = compute_circuit_liveness(
            circuit, benchmark_name, backend_name, count_ff=False)

    metrics['liveness_ff'] = compute_circuit_liveness(
            circuit, benchmark_name, backend_name, count_ff=True)

    metrics['num_gates'] = compute_circuit_num_gates(
            circuit, benchmark_name, backend_name,
            count_measure=False, count_reset=False, count_ff=False)

    metrics['num_gates_measure_reset'] = compute_circuit_num_gates(
            circuit, benchmark_name, backend_name,
            count_measure=True, count_reset=True, count_ff=False)

    metrics['num_gates_measure_reset_ff'] = compute_circuit_num_gates(
            circuit, benchmark_name, backend_name,
            count_measure=True, count_reset=True, count_ff=True)

    metrics['critical_path_quantum'] = compute_circuit_critical_depth_quantum(
            circuit, benchmark_name, backend_name)

    metrics['critical_path_quantum_classical'] = \
            compute_circuit_critical_depth_quantum_classical(
                    circuit, benchmark_name, backend_name)

    metrics['mcm_depth_ratio'] = compute_circuit_mcm_depth_ratio(
            circuit, benchmark_name, backend_name)

    metrics['mcm_depth_ratio_ff'] = compute_circuit_mcm_depth_ratio(
            circuit, benchmark_name, backend_name, count_ff=True)

    metrics['mcm_plus_ff_depth_ratio'] = compute_circuit_mcm_plus_ff_depth_ratio(
            circuit, benchmark_name, backend_name)

    metrics['parallelism'] = compute_circuit_parallelism(
            circuit, benchmark_name, backend_name)

    metrics['parallelism_ff'] = compute_circuit_parallelism(
            circuit, benchmark_name, backend_name, count_ff=True)

    metrics['communication'] = compute_circuit_communication(
            circuit, benchmark_name, backend_name)

    metrics['quantum_entanglement'] = compute_circuit_quantum_entanglement(
            circuit, benchmark_name, backend_name,
            count_measure=False, count_reset=False)

    metrics['quantum_entanglement_measure_reset'] = compute_circuit_quantum_entanglement(
            circuit, benchmark_name, backend_name)

    metrics['quantum_entanglement_measure_reset_ff'] = compute_circuit_quantum_entanglement(
            circuit, benchmark_name, backend_name, count_ff=True)

    metrics['quantum_classical_entanglement'] = compute_circuit_quantum_classical_entanglement(
            circuit, benchmark_name, backend_name,
            count_measure=False, count_reset=False)

    metrics['quantum_classical_entanglement_measure_reset'] = \
            compute_circuit_quantum_classical_entanglement(circuit, benchmark_name, backend_name)

    metrics['quantum_classical_entanglement_measure_reset_ff'] = \
            compute_circuit_quantum_classical_entanglement(
                    circuit, benchmark_name, backend_name, count_ff=True)

    return metrics
