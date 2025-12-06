# =============================================================================
# Maxwell Processor - Docker Build
# =============================================================================
# Multi-stage build for the Complementarity-Aware Processor
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build
# -----------------------------------------------------------------------------
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock* ./
COPY processor/ processor/

# Build release binary
RUN cargo build --release --package maxwell-processor

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 maxwell
USER maxwell

# Copy binary from builder
COPY --from=builder /app/target/release/maxwell /usr/local/bin/maxwell

# Set working directory
WORKDIR /home/maxwell

# Default command
ENTRYPOINT ["maxwell"]
CMD ["--help"]

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
LABEL org.opencontainers.image.title="Maxwell Processor"
LABEL org.opencontainers.image.description="Complementarity-Aware Processor for Categorical Phase-Lock Dynamics"
LABEL org.opencontainers.image.authors="Kundai Farai Sachikonye <kundai.sachikonye@wzw.tum.de>"
LABEL org.opencontainers.image.source="https://github.com/sachikonye/maxwell"
LABEL org.opencontainers.image.licenses="MIT"

