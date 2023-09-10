build:
	cargo build --release

check:
	cargo check
	cargo clippy -- -A clippy::needless_range_loop

fmt:
	cargo fmt

test:
	cargo test

.PHONY: check fmt test build
