.PHONY: setup train optimize benchmark edge demo all clean

setup:
	pip install -r requirements.txt

train:
	python scripts/train_baseline.py

optimize: optimize-quantize optimize-onnx optimize-distill

optimize-quantize:
	python scripts/optimize_quantize.py

optimize-onnx:
	python scripts/optimize_onnx.py

optimize-distill:
	python scripts/distill_to_lstm.py

benchmark:
	python scripts/benchmark.py

edge:
	python scripts/edge_simulate.py

demo:
	python scripts/run_inference.py --model all

all: setup train optimize benchmark edge

clean:
	rm -rf models/ data/ benchmarks/results.json benchmarks/plots/
