build:
	cargo build --release

check:
	cargo check

fmt:
	cargo fmt

test:
	cargo test

.PHONY: check fmt test build
