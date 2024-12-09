prepare:
	cd am205 && ./prepare.sh

devbuild:
	python dev.py build

run-random:
	@if [ -f am205/results/random-$(GEN_FUNC).json ]; then \
		rm am205/results/random-$(GEN_FUNC).json; \
	fi
	@for i in $(shell seq 100 100 2000); do \
		python dev.py -n python am205/random.py -- $$i $(GEN_FUNC); \
	done
	@python dev.py -n python am205/plot_random.py

run-real:
	@if [ -f am205/results/real.json ]; then \
		rm am205/results/real.json; \
	fi
	@for i in $(shell seq 1 33); do \
		python dev.py -n python am205/real.py -- $$i; \
	done
	@python dev.py -n python am205/plot_real.py
