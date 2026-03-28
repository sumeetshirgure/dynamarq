import dynamarq

from qiskit_ibm_runtime import QiskitRuntimeService

print("--- Testing Qiskit metric evaluation benchmarks ---")

service = QiskitRuntimeService()
backend = service.backend('ibm_pittsburgh')
print(backend)

normalized_metrics = [
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

for bm in dynamarq.get_testbench() :
    qiskit_metrics = dynamarq.QiskitMetrics(bm, backend, stretch_dd=False)
    metric_values = qiskit_metrics.get_metrics()
    print(bm.name(), metric_values)
    for normalized_metric_name in normalized_metrics :
        assert 0.0 <= metric_values[normalized_metric_name] <= 1.0 + 1e-6 , f'violation = {metric_values[normalized_metric_name]}'


for bm in dynamarq.get_testbench() :
    qiskit_metrics = dynamarq.QiskitMetrics(bm, backend, stretch_dd=True)
    metric_values = qiskit_metrics.get_metrics()
    print(f"{bm.name()}_dd", metric_values)
    for normalized_metric_name in normalized_metrics :
        assert 0.0 <= metric_values[normalized_metric_name] <= 1.0 + 1e-6 , f'violation = {metric_values[normalized_metric_name]}'
