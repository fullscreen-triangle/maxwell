// Physical constants
export const kB = 1.380649e-23;      // J/K
export const hbar = 1.054571817e-34; // J·s
export const h = 6.62607015e-34;     // J·s
export const c_light = 2.998e8;      // m/s
export const NA = 6.02214076e23;     // 1/mol

// Simulation defaults
export const DEFAULT_PARTICLE_COUNT = 200;
export const DEFAULT_TEMPERATURE = 300.0;    // K
export const DEFAULT_VOLUME = 1.0;           // normalized
export const DEFAULT_PRESSURE = 101325.0;    // Pa

// GPU limits
export const MAX_PARTICLES = 10000;
export const SPECTRAL_RES = 32;  // spectral image NxN
export const VOLUME_RES = 128;   // ray march volume resolution
export const MAX_RAY_STEPS = 128;

// Render
export const MARCH_SIZE = 1.0 / MAX_RAY_STEPS;
