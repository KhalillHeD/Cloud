# Thin wrapper: every target runs `python tasks.py <target>`, which holds the logic and
# also works on Windows without make. Pass extra flags with ARGS, e.g.
#   make check ARGS=--probe-write
PY   ?= python
ARGS ?=

TARGETS := help setup check test lint validate download eda local seed demo train deploy destroy

.PHONY: $(TARGETS)
.DEFAULT_GOAL := help

$(TARGETS):
	$(PY) tasks.py $@ $(ARGS)
