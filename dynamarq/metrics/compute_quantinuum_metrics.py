from ..benchmark import Benchmark

from collections import defaultdict

from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import CircuitInstruction, Qubit

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit


class QuantinuumMetrics() :

    def __init__(self, benchmark: Benchmark) :
        if not isinstance(benchmark, Benchmark) :
            raise ValueError("Benchmark not of type dynamarq.Benchmark")
        self.benchmark = benchmark

        # Median measurement and 2Q error rates taken from here.
        # https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/emulators/helios_emulators.html

        self.median_rx_error = 2.5e-5
        self.median_m2_error = 1e-06
        self.median_2q_error = 8e-4


    def get_metrics(self) :
        circuit_metrics = [self.get_circuit_metrics(circuit)
                           for circuit in self.benchmark.qiskit_circuits(
                               stretch_dd=False)]

        benchmark_metrics = dict()
        for metric_name in self.get_metric_names() :
            metrics_list = [ d[metric_name] for d in circuit_metrics ]
            average_metric = sum(metrics_list) / len(metrics_list)
            benchmark_metrics[metric_name] = average_metric

        return benchmark_metrics


    def get_metric_names(self) :
        return [
                'depth',
                'depth_ff',
                'num_gates',
                'num_gates_measure_reset',
                'num_gates_measure_reset_ff',
                'num_system_qubits',
                'num_total_qubits',
                'liveness',
                'liveness_ff',
                'system_qubit_ratio',
                'critical_path_quantum',
                'critical_path_quantum_classical',
                'mcm_depth_ratio',
                'mcm_plus_ff_depth_ratio',
                'parallelism',
                'parallelism_ff',
                'quantum_communication',
                'quantum_classical_communication',
                'quantum_entanglement',
                'quantum_entanglement_measure_reset',
                'quantum_entanglement_measure_reset_ff',
                'quantum_classical_entanglement',
                'quantum_classical_entanglement_measure_reset',
                'quantum_classical_entanglement_measure_reset_ff',
                ]


    def get_circuit_metrics(self,
                            circuit : QuantumCircuit
                            ) -> dict :
        metrics = dict()

        metrics['depth'] = self.compute_circuit_object_depths(circuit, count_ff=False)[1]

        metrics['depth_ff'] = self.compute_circuit_object_depths(circuit, count_ff=True)[1]

        metrics['num_gates'] = self.compute_circuit_num_gates(
                circuit, count_measure=False, count_reset=False, count_ff=False)

        metrics['num_gates_measure_reset'] = self.compute_circuit_num_gates(
                circuit, count_measure=True, count_reset=True, count_ff=False)

        metrics['num_gates_measure_reset_ff'] = self.compute_circuit_num_gates(
                circuit, count_measure=True, count_reset=True, count_ff=True)

        metrics['num_system_qubits'] = self.compute_circuit_system_qubit_count()

        metrics['num_total_qubits'] = self.compute_circuit_total_qubit_count()

        metrics['liveness'] = self.compute_circuit_liveness(circuit, count_ff=False)

        metrics['liveness_ff'] = self.compute_circuit_liveness(circuit, count_ff=True)

        metrics['system_qubit_ratio'] = self.compute_circuit_system_qubit_ratio()

        metrics['critical_path_quantum'] = self.compute_circuit_critical_depth_quantum(circuit)

        metrics['critical_path_quantum_classical'] = \
                self.compute_circuit_critical_depth_quantum_classical(circuit)

        metrics['mcm_depth_ratio'] = self.compute_circuit_mcm_depth_ratio(circuit)

        metrics['mcm_plus_ff_depth_ratio'] = \
                self.compute_circuit_mcm_plus_ff_depth_ratio(circuit)

        metrics['parallelism'] = self.compute_circuit_parallelism(circuit)

        metrics['parallelism_ff'] = self.compute_circuit_parallelism(circuit, count_ff=True)

        metrics['quantum_communication'] = self.compute_circuit_quantum_communication(circuit)

        metrics['quantum_classical_communication'] = \
                self.compute_circuit_quantum_classical_communication(circuit)

        metrics['quantum_entanglement'] = self.compute_circuit_quantum_entanglement(
                circuit, count_measure=False, count_reset=False)

        metrics['quantum_entanglement_measure_reset'] = \
                self.compute_circuit_quantum_entanglement(circuit)

        metrics['quantum_entanglement_measure_reset_ff'] = \
                self.compute_circuit_quantum_entanglement(circuit, count_ff=True)

        metrics['quantum_classical_entanglement'] = \
                self.compute_circuit_quantum_classical_entanglement(
                        circuit, count_measure=False, count_reset=False)

        metrics['quantum_classical_entanglement_measure_reset'] = \
                self.compute_circuit_quantum_classical_entanglement(circuit)

        metrics['quantum_classical_entanglement_measure_reset_ff'] = \
                self.compute_circuit_quantum_classical_entanglement(circuit, count_ff=True)

        return metrics


    def get_instruction_branch_probability(self, instruction) :
        p, m, s = self.median_2q_error, self.median_m2_error, self.median_rx_error

        if self.benchmark.name() == 'RepetitionCode_3' :
            # Compute most significant bit flip error rate on ancillas.
            bp = 2 * p + m + s
            return bp

        if self.benchmark.name() == 'RepetitionCode_5' :
            # Compute most significant bit flip error rate on ancillas.
            bp = 2 * p + m + s
            return bp

        if self.benchmark.name() == 'FiveQubitCode' :
            # Branch probability calculation based on symmetry.
            if instruction.clbits[0]._register.name == 'syn' :
                return 1.0 / 16.0

        if self.benchmark.name() == 'SteaneCode' :
            bp = 4 * p + m + s
            return bp

        return 0.5


    def get_node_branch_probability(self, node) :
        p, m, s = self.median_2q_error, self.median_m2_error, self.median_rx_error

        if self.benchmark.name() == 'RepetitionCode_3' :
            # Compute most significant bit flip error rate on ancillas.
            bp = 2 * p + m + s
            return bp

        if self.benchmark.name() == 'RepetitionCode_5' :
            # Compute most significant bit flip error rate on ancillas.
            bp = 2 * p + m + s
            return bp

        if self.benchmark.name() == 'FiveQubitCode' :
            # Branch probability calculation based on symmetry.
            if node.cargs[0]._register.name == 'syn' :
                return 1.0 / 16.0

        if self.benchmark.name() == 'SteaneCode' :
            bp = 4 * p + m + s
            return bp

        return 0.5


    def compute_circuit_system_qubit_count(self) -> float :

        if 'GHZ_' in self.benchmark.name() :
            return self.benchmark.n

        if 'GHZReset_' in self.benchmark.name() :
            return self.benchmark.n+1

        if 'CNOTLadder_' in self.benchmark.name() :
            return self.benchmark.n

        if 'Fanout_' in self.benchmark.name() :
            return self.benchmark.n+1

        if 'LongRangeCNOT_' in self.benchmark.name() :
            return 2

        if 'LongRangeCNOTSparse_' in self.benchmark.name() :
            return self.benchmark.n+1

        if 'RepetitionCode_' in self.benchmark.name() :
            return self.benchmark.n

        if self.benchmark.name() == 'FiveQubitCode' :
            return 5

        if 'IPE_' in self.benchmark.name() :
            return 1

        if 'TFIM_' in self.benchmark.name() :
            return self.benchmark.n

        if 'PartialQFT_' in self.benchmark.name() :
            return self.benchmark.n

        if 'QFT_' in self.benchmark.name() :
            return self.benchmark.n

        return 0.0


    def compute_circuit_total_qubit_count(self) -> float :

        if 'GHZ_' in self.benchmark.name() :
            return 2*self.benchmark.n-1

        if 'GHZReset_' in self.benchmark.name() :
            return 2*self.benchmark.n

        if 'CNOTLadder_' in self.benchmark.name() :
            return 2*self.benchmark.n-1

        if 'Fanout_' in self.benchmark.name() :
            return 2*self.benchmark.n+1

        if 'LongRangeCNOT_' in self.benchmark.name() :
            return 2+self.benchmark.n

        if 'LongRangeCNOTSparse_' in self.benchmark.name() :
            return 2*self.benchmark.n+1

        if 'RepetitionCode_' in self.benchmark.name() :
            return 2*self.benchmark.n-1

        if self.benchmark.name() == 'FiveQubitCode' :
            return 10

        if 'IPE_' in self.benchmark.name() :
            return 2

        if 'TFIM_' in self.benchmark.name() :
            return 2*self.benchmark.n-1

        if 'PartialQFT_' in self.benchmark.name() :
            return self.benchmark.n

        if 'QFT_' in self.benchmark.name() :
            return self.benchmark.n

        return 0.0


    def compute_circuit_system_qubit_ratio(self) -> float :

        if 'GHZ_' in self.benchmark.name() :
            return self.benchmark.n / (2*self.benchmark.n-1)

        if 'GHZReset_' in self.benchmark.name() :
            return (self.benchmark.n+1) / (2*self.benchmark.n)

        if 'CNOTLadder_' in self.benchmark.name() :
            return self.benchmark.n / (2*self.benchmark.n-1)

        if 'Fanout_' in self.benchmark.name() :
            return (self.benchmark.n+1) / (2*self.benchmark.n+1)

        if 'LongRangeCNOT_' in self.benchmark.name() :
            return 2 / (2+self.benchmark.n)

        if 'LongRangeCNOTSparse_' in self.benchmark.name() :
            return (self.benchmark.n+1) / (2*self.benchmark.n+1)

        if 'RepetitionCode_' in self.benchmark.name() :
            return self.benchmark.n / (2*self.benchmark.n-1)

        if self.benchmark.name() == 'FiveQubitCode' :
            return 0.5

        if 'IPE_' in self.benchmark.name() :
            return 0.5

        if 'TFIM_' in self.benchmark.name() :
            return self.benchmark.n / (2*self.benchmark.n-1)

        return 1.0


    def compute_circuit_object_depths(self,
                                      circuit : QuantumCircuit,
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

                branch_probability = self.get_instruction_branch_probability(instruction)

                if len(instruction.params) > 0 :
                    _, if_subcircuit_total_depth = self.compute_circuit_object_depths(
                            instruction.params[0])
                else :
                    if_subcircuit_total_depth = 0
                if len(instruction.params) > 1 :
                    _, else_subcircuit_total_depth = self.compute_circuit_object_depths(
                            instruction.params[1])
                else :
                    else_subcircuit_total_depth = 0

                added_depth = branch_probability * if_subcircuit_total_depth + \
                        (1-branch_probability) * else_subcircuit_total_depth

                new_depth += added_depth

            elif instruction.is_standard_gate() or \
                instruction.name in ['reset', 'measure', 'measure_2'] :
                new_depth += 1

            for obj in objects: obj_depths[obj] = new_depth
            total_depth = max(total_depth, new_depth)

        return obj_depths, total_depth


    def compute_circuit_total_active_time(self,
                                          circuit : QuantumCircuit,
                                          count_ff: bool = False,
                                          ) -> float :

        if circuit is None : return 0.0

        qubit_activity = 0.0

        for instruction in circuit :
            if instruction.name in ['barrier', 'measure', 'measure_2'] : continue

            added_activity = 0.0
            if instruction.is_control_flow() and instruction.name == 'if_else' :
                if count_ff : added_activity += 1

                branch_probability = self.get_instruction_branch_probability(instruction)

                if len(instruction.params) > 0 :
                    if_subcircuit_total_activity = \
                            self.compute_circuit_total_active_time(instruction.params[0])
                else :
                    if_subcircuit_total_activity = 0
                if len(instruction.params) > 1 :
                    else_subcircuit_total_activity = \
                            self.compute_circuit_total_active_time(instruction.params[1])
                else :
                    else_subcircuit_total_activity = 0

                added_activity += branch_probability * if_subcircuit_total_activity + \
                        (1 - branch_probability) * else_subcircuit_total_activity

            elif instruction.is_standard_gate() or instruction.name in ['reset'] :
                added_activity += instruction.operation.num_qubits

            qubit_activity += added_activity

        return qubit_activity


    def compute_circuit_liveness(self,
                                 circuit : QuantumCircuit,
                                 count_ff: bool = False,
                                 ) -> float :

        circuit_qubit_depths, _ = \
                self.compute_circuit_object_depths(circuit, count_ff=count_ff)
        total_gate_activity = \
                self.compute_circuit_total_active_time(circuit, count_ff=count_ff)

        total_liveness = 0.0
        for obj, depth in circuit_qubit_depths.items() :
            if isinstance(obj, Qubit) :
                total_liveness += depth

        if total_liveness == 0.0 : return 0.0
        return total_gate_activity / total_liveness


    def compute_circuit_num_gates(self,
                                  circuit : QuantumCircuit,
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

                branch_probability = \
                        self.get_instruction_branch_probability(instruction)

                if len(instruction.params) > 0 :
                    if_subcircuit_total_gates = self.compute_circuit_num_gates(
                            instruction.params[0], count_measure, count_reset, count_ff)
                else :
                    if_subcircuit_total_gates = 0
                if len(instruction.params) > 1 :
                    else_subcircuit_total_gates = self.compute_circuit_num_gates(
                            instruction.params[1], count_measure, count_reset, count_ff)
                else :
                    else_subcircuit_total_gates = 0

                added_num_gates += branch_probability * if_subcircuit_total_gates + \
                        (1 - branch_probability) * else_subcircuit_total_gates

            elif instruction.is_standard_gate() :
                added_num_gates += 1.0

            num_gates += added_num_gates

        return num_gates


    def get_circuit_critical_path(self,
                                  circuit : QuantumCircuit,
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

                    branch_probability = \
                            self.get_node_branch_probability(succ)

                    if len(succ.params) > 0 :
                        _, if_subcircuit_critical_depth = \
                                self.get_circuit_critical_path(succ.params[0], count_ff)
                    else :
                        if_subcircuit_critical_depth = 0
                    if len(succ.params) > 1 :
                        _, else_subcircuit_critical_depth = \
                                self.get_circuit_critical_path(succ.params[1], count_ff)
                    else :
                        else_subcircuit_critical_depth = 0

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


    def compute_classical_entanglement_gates(self,
                                             circuit : QuantumCircuit,
                                             count_measure: bool = True,
                                             count_reset: bool = True,
                                             count_ff: bool = False,
                                             ) -> float:
        num_branch_gates = 0.0

        for instruction in circuit :
            if instruction.is_control_flow() and instruction.name == 'if_else' :

                branch_probability = \
                        self.get_instruction_branch_probability(instruction)

                if len(instruction.params) > 0 :
                    if_subcircuit_total_gates = self.compute_circuit_num_gates(
                            instruction.params[0], count_measure, count_reset, count_ff)
                else :
                    if_subcircuit_total_gates = 0
                if len(instruction.params) > 1 :
                    else_subcircuit_total_gates = self.compute_circuit_num_gates(
                            instruction.params[1], count_measure, count_reset, count_ff)
                else :
                    else_subcircuit_total_gates = 0

                num_branch_gates += branch_probability * if_subcircuit_total_gates + \
                        (1 - branch_probability) * else_subcircuit_total_gates 

        return num_branch_gates


    def compute_circuit_critical_depth_quantum(self,
                                               circuit : QuantumCircuit,
                                               ) -> float :
        critical_path, _ = self.get_circuit_critical_path(circuit, count_ff=False)

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


    def compute_circuit_critical_depth_quantum_classical(self,
                                                         circuit : QuantumCircuit,
                                                         count_measure : bool = True,
                                                         count_reset : bool = True,
                                                         count_ff : bool = False
                                                         ) -> float :
        critical_path, _ = self.get_circuit_critical_path(circuit, count_ff=count_ff)

        num_two_qubit_longest_path = 0
        for node in critical_path :

            if node.op.num_qubits > 1:
                num_two_qubit_longest_path += 1

            elif node.op.name == 'if_else' :
                branch_probability = self.get_node_branch_probability(node)

                if len(node.params) > 0 :
                    if_subcircuit_total_gates = self.compute_circuit_num_gates(
                            node.params[0], count_measure, count_reset, count_ff)
                else :
                    if_subcircuit_total_gates = 0
                if len(node.params) > 1 :
                    else_subcircuit_total_gates = self.compute_circuit_num_gates(
                            node.params[1], count_measure, count_reset, count_ff)
                else :
                    else_subcircuit_total_gates = 0

                num_two_qubit_longest_path += \
                        branch_probability * if_subcircuit_total_gates + \
                        (1 - branch_probability) * else_subcircuit_total_gates

        num_two_qubits_total = 0
        for instruction in circuit :
            if instruction.operation.num_qubits > 1:
                num_two_qubits_total += 1

        num_two_qubits_total += self.compute_classical_entanglement_gates(
                circuit, count_measure, count_reset, count_ff)

        if num_two_qubits_total == 0:
            return 0
        return num_two_qubit_longest_path / num_two_qubits_total


    def compute_circuit_mcm_depth_ratio(self,
                                        circuit : QuantumCircuit,
                                        ) -> float :
        dag = circuit_to_dag(circuit)
        mid_measurement_depth = 0
        total_layers = 0
        for layer in dag.layers():
            total_layers += 1
            layer_ops = layer['graph'].op_nodes()
            for node in layer_ops:
                if node.name == 'measure_2' :
                    mid_measurement_depth += 1
                    break
        if total_layers == 0 :
            return 0
        return mid_measurement_depth / total_layers


    def compute_circuit_mcm_plus_ff_depth_ratio(self,
                                                circuit : QuantumCircuit,
                                                ) -> float :
        dag = circuit_to_dag(circuit)
        total_layers = 0
        mcm_ff_depth = 0
        for layer in dag.layers():
            total_layers += 1
            layer_ops = layer['graph'].op_nodes()
            for node in layer_ops:
                if node.name == 'measure_2' :
                    mcm_ff_depth += 1
                    break
                elif node.is_control_flow() and node.op.name == 'if_else' :
                    mcm_ff_depth += 1
                    break
        if total_layers == 0 :
            return 0
        return mcm_ff_depth / total_layers


    def compute_circuit_parallelism(self,
                                    circuit : QuantumCircuit,
                                    count_ff: bool = False
                                    ) -> float:
        _, depth = \
                self.compute_circuit_object_depths(circuit, count_ff=count_ff)
        num_gates = \
                self.compute_circuit_num_gates(circuit, count_ff=count_ff)
        if circuit.num_qubits <= 1 : return 0
        return max((num_gates / depth - 1) / (circuit.num_qubits - 1), 0)


    def get_quantum_connectivity_graph(self,
                                       circuit : QuantumCircuit,
                                       conditioned_qubits : list = [],
                                       ) -> defaultdict :
        dag = circuit_to_dag(circuit)
        dag.remove_all_ops_named('barrier')
        graph = set()
        for node in dag.topological_op_nodes() :
            if node.is_standard_gate() :
                for conditioned_qubit in conditioned_qubits :
                    for gate_qubit in node.qargs :
                        graph.add((conditioned_qubit, gate_qubit))
                        graph.add((gate_qubit, conditioned_qubit))
                if node.op.num_qubits == 2 :
                    graph.add((node.qargs[0], node.qargs[1]))
                    graph.add((node.qargs[1], node.qargs[0]))
        return graph


    def compute_circuit_quantum_communication(self,
                                              circuit : QuantumCircuit,
                                              ) -> float :
        num_qubits = circuit.num_qubits
        if num_qubits <= 1 : return 0
        graph = self.get_quantum_connectivity_graph(circuit)
        degree_sum = len(graph)
        return degree_sum / (num_qubits * (num_qubits - 1))


    def compute_circuit_quantum_classical_communication(self,
                                                        circuit : QuantumCircuit,
                                                        ) -> float:
        num_qubits = circuit.num_qubits
        dag = circuit_to_dag(circuit)
        if num_qubits <= 1 : return 0
        dag.remove_all_ops_named('barrier')
        graph = defaultdict(list)
        clbit_map = dict()
        for node in dag.topological_op_nodes() :
            if node.op.name in ['measure', 'measure_2'] :
                clbit_map[node.cargs[0]] = node.qargs[0]
                continue
            if node.is_standard_gate() and node.op.num_qubits == 2 :
                graph[(node.qargs[0], node.qargs[1])] += [1.0]
                graph[(node.qargs[1], node.qargs[0])] += [1.0]
                continue
            if node.is_control_flow() and node.op.name == 'if_else' :
                p = self.get_node_branch_probability(node)
                subgraph = defaultdict(int)
                conditioned_bits = list()
                for carg in node.cargs :
                    if carg in clbit_map :
                        conditioned_bits.append(clbit_map[carg]) 
                if len(node.params) > 0 :
                    if_subcircuit_subgraph = self.get_quantum_connectivity_graph(
                            node.params[0], conditioned_bits)
                    for q0, q1 in if_subcircuit_subgraph :
                        subgraph[(q0, q1)] += p
                        subgraph[(q1, q0)] += p
                if len(node.params) > 1 :
                    else_subcircuit_subgraph = self.get_quantum_connectivity_graph(
                            node.params[1], conditioned_bits)
                    for q1, q0 in else_subcircuit_subgraph :
                        subgraph[(q0, q1)] += 1-p
                        subgraph[(q1, q0)] += 1-p
                for edge, value in subgraph.items() :
                    graph[edge] += [value]
        communication = 0
        for edge, instruction_probabilities in graph.items() :
            if edge[0] == edge[1] : continue
            product = 1
            for instruction_probability in instruction_probabilities :
                product *= (1 - instruction_probability)
            communication += 1 - product
        return communication / (num_qubits * (num_qubits - 1))


    def compute_circuit_quantum_entanglement(self,
                                             circuit : QuantumCircuit,
                                             count_measure : bool = True,
                                             count_reset : bool = True,
                                             count_ff : bool = False
                                             ) -> float :
        num_two_qubit_gates = 0
        num_gates = self.compute_circuit_num_gates(
                circuit, count_measure, count_reset, count_ff)
        for instruction in circuit :
            if instruction.is_standard_gate() and instruction.operation.num_qubits == 2 :
                num_two_qubit_gates += 1
        if num_gates == 0 : return 0
        return num_two_qubit_gates / num_gates


    def compute_circuit_quantum_classical_entanglement(self,
                                                       circuit : QuantumCircuit,
                                                       count_measure : bool = True,
                                                       count_reset : bool = True,
                                                       count_ff : bool = False
                                                       ) -> float :
        quantum_entanglement = self.compute_circuit_quantum_entanglement(
                circuit, count_measure, count_reset, count_ff)
        num_gates = self.compute_circuit_num_gates(
                circuit, count_measure, count_reset, count_ff)
        classical_entanglement_gates = self.compute_classical_entanglement_gates(
                circuit, count_measure, count_reset, count_ff)
        return quantum_entanglement + classical_entanglement_gates / num_gates

