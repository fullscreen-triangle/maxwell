# =============================================================================
# Maxwell Project - Makefile
# =============================================================================
# Complementarity-Aware Processor for Categorical Phase-Lock Dynamics
# =============================================================================

.PHONY: all build release test bench clean docs lint fmt check install \
        demo dissolution run-validation docker help

# Default target
all: build test

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

build:
	@echo "🔨 Building Maxwell processor..."
	cargo build --workspace

release:
	@echo "🚀 Building release..."
	cargo build --workspace --release

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

test:
	@echo "🧪 Running Rust tests..."
	cargo test --workspace

test-verbose:
	@echo "🧪 Running Rust tests (verbose)..."
	cargo test --workspace -- --nocapture

bench:
	@echo "📊 Running benchmarks..."
	cargo bench --workspace

# -----------------------------------------------------------------------------
# Python Validation
# -----------------------------------------------------------------------------

install-validation:
	@echo "📦 Installing Python validation package..."
	cd validation && pip install -e ".[dev]"

run-validation:
	@echo "🔬 Running Python validation suite..."
	cd validation && python -m maxwell_validation.dissolution

validate-all: install-validation run-validation

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------

lint:
	@echo "🔍 Running clippy..."
	cargo clippy --workspace --all-targets -- -D warnings

fmt:
	@echo "✨ Formatting code..."
	cargo fmt --all

fmt-check:
	@echo "🔍 Checking formatting..."
	cargo fmt --all -- --check

check: fmt-check lint test
	@echo "✅ All checks passed!"

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------

docs:
	@echo "📚 Building documentation..."
	cargo doc --workspace --no-deps --open

docs-private:
	@echo "📚 Building documentation (including private items)..."
	cargo doc --workspace --no-deps --document-private-items --open

# -----------------------------------------------------------------------------
# Demos
# -----------------------------------------------------------------------------

demo:
	@echo "🎮 Running demo..."
	cargo run --release -- demo --molecules 1000 --temperature 300

dissolution:
	@echo "👻 Demonstrating the seven-fold dissolution..."
	cargo run --release -- dissolution

complementarity:
	@echo "🔄 Showing complementarity..."
	cargo run --release -- complementarity

retrieval-paradox:
	@echo "🔁 Running retrieval paradox demo..."
	cargo run --release -- retrieval-paradox --steps 100

projection:
	@echo "📽️ Showing projection..."
	cargo run --release -- projection

complete:
	@echo "🌳 Running 3^k completion..."
	cargo run --release -- complete --depth 5

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t maxwell-processor:latest .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run --rm maxwell-processor:latest dissolution

docker-compose-up:
	@echo "🐳 Starting services..."
	docker-compose up -d

docker-compose-down:
	@echo "🐳 Stopping services..."
	docker-compose down

# -----------------------------------------------------------------------------
# LaTeX Documents
# -----------------------------------------------------------------------------

docs-latex:
	@echo "📄 Building LaTeX documents..."
	cd docs/resolution && latexmk -pdf resolution-of-maxwell-demons.tex

docs-latex-clean:
	@echo "🧹 Cleaning LaTeX artifacts..."
	cd docs/resolution && latexmk -C

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------

clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean docs-latex-clean
	@echo "🧹 Deep clean complete!"

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

install:
	@echo "📦 Installing Maxwell processor..."
	cargo install --path processor

uninstall:
	@echo "🗑️ Uninstalling Maxwell processor..."
	cargo uninstall maxwell-processor

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

dev:
	@echo "🔧 Setting up development environment..."
	rustup component add clippy rustfmt
	$(MAKE) install-validation

watch:
	@echo "👀 Watching for changes..."
	cargo watch -x check -x test

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help:
	@echo "Maxwell Project - Available Commands"
	@echo "====================================="
	@echo ""
	@echo "Build:"
	@echo "  make build          - Build debug version"
	@echo "  make release        - Build release version"
	@echo ""
	@echo "Test:"
	@echo "  make test           - Run all Rust tests"
	@echo "  make bench          - Run benchmarks"
	@echo "  make run-validation - Run Python validation"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run clippy"
	@echo "  make fmt            - Format code"
	@echo "  make check          - Run all checks"
	@echo ""
	@echo "Demos:"
	@echo "  make demo           - Run processor demo"
	@echo "  make dissolution    - Show seven-fold dissolution"
	@echo "  make complementarity - Show face complementarity"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Build Rust documentation"
	@echo "  make docs-latex     - Build LaTeX papers"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run in Docker"
	@echo ""
	@echo "Other:"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make install        - Install processor binary"
	@echo "  make dev            - Setup development environment"

