run-tmp:
	cargo run -- --dev --tmp -lruntime=debug

purge:
	cargo run -- purge-chain --dev -y

restart: purge run

.PHONY: init
init:
	./scripts/init.sh

.PHONY: check
check:
	SKIP_WASM_BUILD=1 cargo check --release

.PHONY: test
test:
	SKIP_WASM_BUILD=1 cargo test --release --all

.PHONY: run
run:
	cargo run --release -- --dev --tmp

.PHONY: build
build:
	cargo build --release

.PHONY: benchmark
benchmark:
	cargo run --manifest-path node/Cargo.toml --features runtime-benchmarks -- benchmark --extrinsic '*' --pallet '*'

.PHONY: benchmark-output
benchmark-output:
	cd runtime/src && cargo run --manifest-path ../../node/Cargo.toml --features runtime-benchmarks -- benchmark --extrinsic '*' --pallet pallet_kitties --template ../../frame-weight-template.hbs --output ../../pallets/kitties/src/weights.rs

.PHONY: test-benchmark
test-benchmark:
	cargo test --manifest-path pallets/kitties/Cargo.toml --features runtime-benchmarks -- --nocapture

