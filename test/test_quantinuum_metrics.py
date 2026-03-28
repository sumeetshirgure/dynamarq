import dynamarq

print("--- Testing Quantinuum metric evaluation benchmarks ---")

all_benchmarks = dynamarq.get_testbench()

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

guppy_benchmarks = []
for bm in all_benchmarks:
    guppy_benchmarks.append(bm)


for bm in guppy_benchmarks :
    quantinuum_metrics = dynamarq.QuantinuumMetrics(bm)
    metric_values = quantinuum_metrics.get_metrics()
    print(bm.name(), metric_values)
    for normalized_metric_name in normalized_metrics :
        assert 0.0 <= metric_values[normalized_metric_name] <= 1.0 + 1e-6 , f'{normalized_metric_name} : violation = {metric_values[normalized_metric_name]}'
