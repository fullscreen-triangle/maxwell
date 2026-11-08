export const PRESETS = {
  'ideal-gas-300k': {
    name: 'Ideal Gas (300K)',
    mode: 'gas',
    particles: 200,
    temperature: 300,
    volume: 1.0,
    molecularMass: 4.65e-26, // N2
    gamma: 1.4,
  },
  'monatomic-500k': {
    name: 'Monatomic Gas (500K)',
    mode: 'gas',
    particles: 200,
    temperature: 500,
    volume: 1.0,
    molecularMass: 6.63e-26, // Ar
    gamma: 5 / 3,
  },
  'water-293k': {
    name: 'Water (20°C)',
    mode: 'fluid',
    particles: 500,
    temperature: 293,
    volume: 0.3,
    molecularMass: 2.99e-26, // H2O
    tau_c: 0.15e-12,
    g_coupling: 6.6,
    viscosity: 1.0e-3,
  },
  'ethanol-293k': {
    name: 'Ethanol (20°C)',
    mode: 'fluid',
    particles: 500,
    temperature: 293,
    volume: 0.3,
    molecularMass: 7.64e-26,
    tau_c: 0.22e-12,
    g_coupling: 5.1,
    viscosity: 1.07e-3,
  },
  'glycerol-293k': {
    name: 'Glycerol (20°C)',
    mode: 'fluid',
    particles: 500,
    temperature: 293,
    volume: 0.2,
    molecularMass: 1.53e-25,
    tau_c: 2.8e-12,
    g_coupling: 334,
    viscosity: 0.934,
  },
};
