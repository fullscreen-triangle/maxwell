//! Benchmarks for the Maxwell Processor
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use maxwell_processor::*;

/// Benchmark categorical state creation
fn bench_state_creation(c: &mut Criterion) {
    let engine = CategoricalEngine::new();
    
    c.bench_function("create_state", |b| {
        b.iter(|| {
            let coords = SCoordinates::new(
                black_box(0.5),
                black_box(0.3),
                black_box(0.7),
            );
            engine.create_state(coords)
        })
    });
}

/// Benchmark phase-lock formation
fn bench_phase_lock_formation(c: &mut Criterion) {
    let engine = CategoricalEngine::new();
    
    // Pre-create states
    for i in 0..100 {
        let coords = SCoordinates::new(i as f64 * 0.01, 0.0, 0.0);
        engine.create_state(coords);
    }
    
    c.bench_function("form_phase_lock", |b| {
        let mut i = 0;
        b.iter(|| {
            let _ = engine.form_phase_lock(
                black_box(i % 100),
                black_box((i + 1) % 100),
                black_box(1.0),
            );
            i += 1;
        })
    });
}

/// Benchmark network construction at various scales
fn bench_network_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_construction");
    
    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                b.iter(|| {
                    let engine = CategoricalEngine::new();
                    let positions: Vec<[f64; 3]> = (0..size)
                        .map(|i| [i as f64 * 1e-10, 0.0, 0.0])
                        .collect();
                    let types = vec![MolecularType::NonPolar; size];
                    engine.construct_network(&positions, &types)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Maxwell-Boltzmann initialization
fn bench_maxwell_boltzmann(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxwell_boltzmann");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let engine = KineticEngine::new();
                engine.set_temperature(300.0);
                
                b.iter(|| {
                    engine.initialize_maxwell_boltzmann(black_box(size), black_box(1.0))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark demon sorting
fn bench_demon_sorting(c: &mut Criterion) {
    let engine = KineticEngine::new();
    engine.set_temperature(300.0);
    engine.initialize_maxwell_boltzmann(1000, 1.0);
    
    c.bench_function("demon_sorting", |b| {
        b.iter(|| {
            engine.demon_sorting()
        })
    });
}

/// Benchmark recursive completion
fn bench_recursive_completion(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_completion");
    
    for depth in [2, 3, 4, 5].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(depth),
            depth,
            |b, &depth| {
                let engine = RecursiveCompletionEngine::new();
                let initial = CategoricalState::new(0, SCoordinates::new(0.0, 0.0, 0.0));
                
                b.iter(|| {
                    engine.complete(black_box(&initial), black_box(depth))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark face switching
fn bench_face_switching(c: &mut Criterion) {
    let processor = MaxwellProcessor::new();
    
    c.bench_function("face_switching", |b| {
        b.iter(|| {
            processor.switch_face()
        })
    });
}

/// Benchmark projection
fn bench_projection(c: &mut Criterion) {
    let explainer = ProjectionExplainer::new();
    let state = CategoricalState::new(42, SCoordinates::new(0.5, 0.3, 0.7));
    
    c.bench_function("project_to_kinetic", |b| {
        b.iter(|| {
            explainer.project(black_box(&state))
        })
    });
}

/// Benchmark equivalence filtering
fn bench_equivalence_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("equivalence_filter");
    
    for size in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let filter = EquivalenceFilter::new();
                let states: Vec<CategoricalState> = (0..size)
                    .map(|i| {
                        CategoricalState::new(
                            i as u64,
                            SCoordinates::new(
                                (i as f64 * 0.05).floor() * 0.05,
                                0.0,
                                0.0,
                            ),
                        )
                    })
                    .collect();
                
                b.iter(|| {
                    filter.filter(black_box(&states))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_state_creation,
    bench_phase_lock_formation,
    bench_network_construction,
    bench_maxwell_boltzmann,
    bench_demon_sorting,
    bench_recursive_completion,
    bench_face_switching,
    bench_projection,
    bench_equivalence_filter,
);

criterion_main!(benches);

