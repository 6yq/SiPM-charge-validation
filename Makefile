VENV:=/mnt/stage/liuyq/tao/venv
PYTHON:=$(VENV)/bin/python
# fitter is a git submodule at ./fitter; add project root so 'import fitter' resolves
FITTER_DIR:=$(CURDIR)
# jax_dep installs JAX related packages.
JAX_DEP:=/mnt/stage/liuyq/tao/jax_dep
export PYTHONPATH:=$(FITTER_DIR):$(JAX_DEP)
# Keep CPU JAX fits from spawning hundreds of worker threads on large hosts.
export XLA_FLAGS:=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8 --xla_cpu_parallel_codegen_split_count=1
export OMP_NUM_THREADS:=8
export OPENBLAS_NUM_THREADS:=8
export MKL_NUM_THREADS:=8
export NUMEXPR_NUM_THREADS:=8

VSTEMS:=53 53.5 54 54.5 55 55.5 56 56.5 57 57.5 58 58.5 59 59.5 60
SEEDS:=16
DATA_FILES:=$(VSTEMS:%=data/PCB6_MPPC_%V_histo.txt)
RESULT_FILES:=$(VSTEMS:%=results/PCB6_MPPC_%V.json)

.PHONY: all data fit scan validate clean

all: results/fit_results.csv

# ─── data ────────────────────────────────────────────────────────────────────

data: .data.stamp

.data.stamp: scripts/download.py
	mkdir -p data
	$(PYTHON) $< -o data/
	touch $@

$(DATA_FILES): .data.stamp

# ─── fit ─────────────────────────────────────────────────────────────────────

fit: results/fit_results.csv

results/PCB6_MPPC_%V.json: data/PCB6_MPPC_%V_histo.txt
	mkdir -p results figures
	{ time $(PYTHON) scripts/fit.py $< -o $@ \
		--out-fig figures/PCB6_MPPC_$*V.pdf \
		--voltage $* \
		--n-seeds $(SEEDS) --maxiter 1000 \
		--dcr-auto ; } 1>$@.log 2>$@.time.log

# ─── validate ────────────────────────────────────────────────────────────────

validate: results/fit_results.csv

results/fit_results.csv: $(DATA_FILES) scripts/model_pipeline.py scripts/fit.py scripts/validate.py
	mkdir -p figures
	$(PYTHON) scripts/model_pipeline.py \
		--seeds $(SEEDS) --maxiter 1000 \
		--out-csv results/fit_results.csv \
		--out-model-csv results/fit_results_models.csv \
		--out-val figures/validation.pdf

# ─── clean ───────────────────────────────────────────────────────────────────

clean:
	rm -rf data results figures .data.stamp
