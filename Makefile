# End-to-end reproducible pipeline.  `make all` = data -> train -> export -> evaluate -> benchmark.
PY ?= python
PRIMARY ?= efficientnet_b0
ALT ?= mobilenetv3_large_100

.PHONY: help install data weights prepare train train-alt export evaluate benchmark examples test lint serve \
        docker-build docker-run all clean

help:          ## show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n",$$1,$$2}'

install:       ## install training + dev dependencies (editable)
	$(PY) -m pip install -r requirements/dev.txt && $(PY) -m pip install --no-deps -e .

data:          ## download the casting dataset to data/raw
	./scripts/download_data.sh data/raw

weights:       ## download ImageNet weights to weights/
	./scripts/download_weights.sh weights

prepare: data  ## EDA + near-duplicate-aware stratified split -> data/splits.csv, reports/data_analysis.json
	$(PY) -m defect_detection.data.prepare --raw-dir data/raw --out data/splits.csv

train: weights ## train the primary model (EfficientNet-B0)
	$(PY) -m defect_detection.training.train --config configs/efficientnet_b0.yaml

train-alt: weights ## train the latency-optimised alternative (MobileNetV3-Large)
	$(PY) -m defect_detection.training.train --config configs/mobilenetv3_large.yaml

export:        ## export both models to ONNX (+INT8) in models/
	$(PY) -m defect_detection.export.onnx_export --checkpoint runs/$(PRIMARY)/best.pt --out-dir models/$(PRIMARY) --int8
	$(PY) -m defect_detection.export.onnx_export --checkpoint runs/$(ALT)/best.pt --out-dir models/$(ALT) --int8

evaluate:      ## test-set evaluation + error analysis -> reports/<model>/
	$(PY) -m defect_detection.evaluation.evaluate --model-dir models/$(PRIMARY) --checkpoint runs/$(PRIMARY)/best.pt --out-dir reports/$(PRIMARY)
	$(PY) -m defect_detection.evaluation.evaluate --model-dir models/$(ALT) --checkpoint runs/$(ALT)/best.pt --out-dir reports/$(ALT)

benchmark:     ## latency / size / accuracy trade-off table -> reports/benchmark.{json,md}
	$(PY) -m defect_detection.evaluation.benchmark \
	  --model $(PRIMARY)=models/$(PRIMARY):runs/$(PRIMARY)/best.pt \
	  --model $(ALT)=models/$(ALT):runs/$(ALT)/best.pt --out reports/benchmark.json

examples:      ## regenerate examples/ through the API
	$(PY) scripts/make_examples.py --model-dir models/$(PRIMARY)

test:          ## unit + API tests
	$(PY) -m pytest

lint:
	ruff check src tests scripts

serve:         ## run the API locally on :8000
	DD_MODEL_DIR=models/$(PRIMARY) DD_LOG_JSON=false uvicorn defect_detection.api.main:app --host 0.0.0.0 --port 8000

docker-build:  ## build the production image
	docker build -f docker/Dockerfile -t defect-detection-api:1.0.0 .

docker-run:    ## run the production image on :8000
	docker run --rm -p 8000:8000 defect-detection-api:1.0.0

all: prepare train train-alt export evaluate benchmark examples

clean:
	rm -rf runs .pytest_cache
