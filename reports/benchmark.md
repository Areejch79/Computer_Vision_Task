Environment: {'cpu': 'x86_64', 'cpu_count': 4, 'threads': 4, 'python': '3.11.15'} (Intel Xeon @ 2.10GHz, 4 vCPU, AVX-512 + VNNI + AMX-INT8)

| model | runtime | size_mb | preprocess_ms_p50 | batch1_p50_ms | batch1_p95_ms | batch8_per_image_ms | throughput_img_s_batch8 | test_f1 | test_recall | test_precision | test_fn | test_fp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | pytorch-fp32 (eager) | 16.21 | 1.94 | 17.31 | 24.55 | 17.3 | 57.8 | – | – | – | – | – |
| efficientnet_b0 | onnx-fp32 | 16.46 | 1.69 | 4.81 | 5.25 | 3.67 | 272.5 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| efficientnet_b0 | onnx-int8 | 5.11 | 1.94 | 7.3 | 8.27 | 5.22 | 191.6 | 0.7778 | 0.9492 | 0.6588 | 6 | 58 |
| mobilenetv3_large_100 | pytorch-fp32 (eager) | 16.92 | 2.02 | 11.13 | 13.02 | 7.61 | 131.4 | – | – | – | – | – |
| mobilenetv3_large_100 | onnx-fp32 | 17.08 | 1.79 | 2.58 | 2.82 | 1.62 | 616.3 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| mobilenetv3_large_100 | onnx-int8 | 4.63 | 2.02 | 3.8 | 4.39 | 2.58 | 388.0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
