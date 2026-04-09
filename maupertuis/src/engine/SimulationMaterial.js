import * as THREE from 'three';

// GPGPU simulation shader — runs as fragment shader on FBO
// Each texel = one particle, RGBA = (Sk, St, Se, speed)
// Second texture: RGBA = (vx, vy, vz, partition_n)

const simulationVertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const simulationFragmentShader = /* glsl */ `
uniform sampler2D positions;   // current S-entropy positions (Sk, St, Se, speed)
uniform sampler2D velocities;  // current velocities (vx, vy, vz, n_level)
uniform float uTime;
uniform float uDeltaTime;
uniform float uTemperature;    // K
uniform float uVolume;         // normalized [0,1]
uniform float uParticleCount;
uniform float uNetworkDensity; // 0=gas, 1=liquid
uniform float uCouplingRadius;
uniform vec2 uResolution;      // FBO resolution

varying vec2 vUv;

// Hash-based random
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 hash3(vec2 p) {
  return vec3(
    hash(p),
    hash(p + vec2(37.0, 17.0)),
    hash(p + vec2(59.0, 83.0))
  );
}

// S-entropy distance
float sDistance(vec3 a, vec3 b) {
  return length(a - b);
}

// Maxwell-Boltzmann speed sampling (Box-Muller approximation)
float mbSpeed(float temperature, float mass, vec2 seed) {
  float sigma = sqrt(temperature * 1.380649e-23 / mass) * 1e-4; // scaled
  float u1 = max(hash(seed), 1e-6);
  float u2 = hash(seed + vec2(0.1, 0.2));
  return sigma * sqrt(-2.0 * log(u1)) * cos(6.28318 * u2);
}

void main() {
  vec4 pos = texture2D(positions, vUv);
  vec4 vel = texture2D(velocities, vUv);

  vec3 sCoord = pos.xyz;   // (Sk, St, Se)
  float speed = pos.w;
  vec3 v = vel.xyz;
  float n_level = vel.w;

  // Unique seed per particle per frame
  vec2 seed = vUv * 1000.0 + vec2(uTime * 7.31, uTime * 13.17);

  // --- KINETIC STEP ---
  // Velocity perturbation (thermal fluctuation)
  float thermalKick = 0.0002 * sqrt(uTemperature / 300.0);
  v += (hash3(seed) - 0.5) * thermalKick;

  // Network density affects damping (gas=free, liquid=viscous)
  float damping = 1.0 - uNetworkDensity * 0.02;
  v *= damping;

  // Update S-coordinates
  sCoord += v * uDeltaTime;

  // Boundary reflection (container walls at margins of [0,1]³)
  float margin = 0.02;
  float upperBound = uVolume;
  for (int d = 0; d < 3; d++) {
    if (sCoord[d] < margin) {
      sCoord[d] = 2.0 * margin - sCoord[d];
      v[d] = abs(v[d]);
    }
    if (sCoord[d] > upperBound - margin) {
      sCoord[d] = 2.0 * (upperBound - margin) - sCoord[d];
      v[d] = -abs(v[d]);
    }
  }

  // Clamp to valid range
  sCoord = clamp(sCoord, vec3(margin), vec3(upperBound - margin));

  // Speed from velocity magnitude
  speed = length(v);

  // Partition level from Sk (principal quantum number proxy)
  n_level = floor(sCoord.x * 5.0) + 1.0;

  gl_FragColor = vec4(sCoord, speed);
}
`;

export class SimulationMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      vertexShader: simulationVertexShader,
      fragmentShader: simulationFragmentShader,
      uniforms: {
        positions: { value: null },
        velocities: { value: null },
        uTime: { value: 0 },
        uDeltaTime: { value: 0.016 },
        uTemperature: { value: 300.0 },
        uVolume: { value: 1.0 },
        uParticleCount: { value: 200 },
        uNetworkDensity: { value: 0.0 },
        uCouplingRadius: { value: 0.1 },
        uResolution: { value: new THREE.Vector2(256, 256) },
      },
    });
  }
}
