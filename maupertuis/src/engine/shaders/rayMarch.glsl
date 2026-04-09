// ═══════════════════════════════════════════════════════════
// Spectral-Native Ray March with Looping Resonator + RPI
// Derived light: c = Δx/τ_c, E = ℏω
// Triple observation: optical + kinetic + thermodynamic
// ═══════════════════════════════════════════════════════════

uniform sampler2D uPositions;     // particle S-coords (Sk, St, Se, speed)
uniform sampler2D uVelocities;    // particle velocities (vx, vy, vz, n)
uniform float uTime;
uniform vec2 uResolution;
uniform float uTemperature;
uniform float uVolume;
uniform float uParticleCount;
uniform float uNetworkDensity;    // 0=gas, 1=liquid
uniform vec3 uCameraPos;
uniform mat4 uCameraMatrix;

varying vec2 vUv;

#define PI 3.14159265359
#define MAX_STEPS 96
#define MAX_PARTICLES_SAMPLE 64
#define ABSORPTION_COEFF 0.9
#define SCATTERING_G 0.3
#define MARCH_SIZE 0.015

// ─── Noise ────────────────────────────────────────────────
float hash(vec3 p) {
  p = fract(p * vec3(443.897, 441.423, 437.195));
  p += dot(p, p.yzx + 19.19);
  return fract((p.x + p.y) * p.z);
}

float noise3D(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x),
        mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
    mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
        mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
    f.z
  );
}

float fbm(vec3 p) {
  float val = 0.0;
  float amp = 0.5;
  float freq = 1.0;
  for (int i = 0; i < 4; i++) {
    val += amp * noise3D(p * freq);
    freq *= 2.02;
    amp *= 0.5;
  }
  return val;
}

// ─── Henyey-Greenstein phase function ─────────────────────
float HenyeyGreenstein(float g, float cosTheta) {
  float gg = g * g;
  return (1.0 / (4.0 * PI)) * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * cosTheta, 1.5));
}

// ─── Beer-Lambert absorption ──────────────────────────────
float BeersLaw(float dist, float absorption) {
  return exp(-dist * absorption);
}

// ─── Cosine color palette ─────────────────────────────────
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  return a + b * cos(6.28318 * (c * t + d));
}

// ─── Sample particle density at position ──────────────────
// Reads from the FBO position texture to find nearby particles
float sampleDensity(vec3 pos) {
  float density = 0.0;
  float texSize = sqrt(uParticleCount);
  float invTex = 1.0 / texSize;

  // Sample a subset of particles for density estimation
  for (int i = 0; i < MAX_PARTICLES_SAMPLE; i++) {
    float fi = float(i);
    vec2 puv = vec2(
      mod(fi, texSize) * invTex + invTex * 0.5,
      floor(fi / texSize) * invTex + invTex * 0.5
    );
    vec4 pData = texture2D(uPositions, puv);
    vec3 pPos = pData.xyz * uVolume; // scale to volume

    float d = length(pos - pPos);
    // Smooth kernel (Gaussian-ish)
    float r = 0.06 + uNetworkDensity * 0.04; // wider in liquid
    density += exp(-d * d / (2.0 * r * r));
  }

  return density;
}

// ─── Sample partition state at position ───────────────────
vec4 samplePartitionState(vec3 pos) {
  float minDist = 999.0;
  vec4 nearest = vec4(0.5, 0.5, 0.5, 0.0);

  float texSize = sqrt(uParticleCount);
  float invTex = 1.0 / texSize;

  for (int i = 0; i < MAX_PARTICLES_SAMPLE; i++) {
    float fi = float(i);
    vec2 puv = vec2(
      mod(fi, texSize) * invTex + invTex * 0.5,
      floor(fi / texSize) * invTex + invTex * 0.5
    );
    vec4 pData = texture2D(uPositions, puv);
    vec3 pPos = pData.xyz * uVolume;

    float d = length(pos - pPos);
    if (d < minDist) {
      minDist = d;
      nearest = pData;
    }
  }

  return nearest;
}

// ─── Local temperature from kinetic energy ────────────────
float localTemperature(vec3 pos) {
  float keSum = 0.0;
  float count = 0.0;
  float texSize = sqrt(uParticleCount);
  float invTex = 1.0 / texSize;

  for (int i = 0; i < MAX_PARTICLES_SAMPLE; i++) {
    float fi = float(i);
    vec2 puv = vec2(
      mod(fi, texSize) * invTex + invTex * 0.5,
      floor(fi / texSize) * invTex + invTex * 0.5
    );
    vec4 pData = texture2D(uPositions, puv);
    vec3 pPos = pData.xyz * uVolume;
    float speed = pData.w;

    float d = length(pos - pPos);
    float w = exp(-d * d / 0.02);
    keSum += w * speed * speed;
    count += w;
  }

  return (count > 0.01) ? keSum / count : 0.0;
}

// ─── Refractive index from network density (RPI) ─────────
float refractiveIndex(vec3 pos) {
  float density = sampleDensity(pos);
  // Gas: n ≈ 1.0, Liquid: n ≈ 1.33-1.5
  return 1.0 + 0.33 * smoothstep(0.0, 5.0, density) * uNetworkDensity;
}

// ─── Main ray march ───────────────────────────────────────
void main() {
  // Camera ray setup
  vec2 uv = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
  uv.x *= uResolution.x / uResolution.y;

  vec3 ro = uCameraPos;
  vec3 rd = normalize(vec3(uv, -1.5));
  // Apply camera rotation
  rd = (uCameraMatrix * vec4(rd, 0.0)).xyz;

  // ── Ray march through volume ──
  vec3 color = vec3(0.0);
  float transmittance = 1.0;
  float phaseAccum = 0.0;
  float totalDensity = 0.0;

  // Stochastic offset (temporal dithering)
  float offset = hash(vec3(gl_FragCoord.xy, uTime)) * MARCH_SIZE;

  vec3 sunDir = normalize(vec3(0.5, 1.0, 0.3));
  vec3 sunColor = vec3(1.0, 0.95, 0.85);

  for (int i = 0; i < MAX_STEPS; i++) {
    float t = float(i) * MARCH_SIZE + offset;
    vec3 pos = ro + rd * t;

    // Check volume bounds [0, uVolume]³
    if (any(lessThan(pos, vec3(0.0))) || any(greaterThan(pos, vec3(uVolume)))) {
      continue;
    }

    // ── SAMPLE LOCAL STATE ──
    float density = sampleDensity(pos);
    vec4 partState = samplePartitionState(pos);
    float n_level = floor(partState.x * 5.0) + 1.0;
    float localT = localTemperature(pos);
    float localSpeed = partState.w;

    // ── OPTICAL: Beer-Lambert with partition-determined μ_abs ──
    float mu_abs = ABSORPTION_COEFF * (n_level / 5.0) * density * 0.1;
    transmittance *= BeersLaw(MARCH_SIZE, mu_abs);

    // ── SCATTERING ──
    float cosTheta = dot(rd, sunDir);
    float scatter;
    if (uNetworkDensity < 0.3) {
      // Gas: Rayleigh (∝ λ⁻⁴, ∝ n⁴)
      scatter = 0.02 * pow(n_level / 5.0, 4.0) * density;
    } else {
      // Liquid: Mie (broader, stronger)
      scatter = 0.08 * pow(n_level / 5.0, 2.0) * density;
    }
    float phase = HenyeyGreenstein(SCATTERING_G, cosTheta);

    // ── THERMAL EMISSION ──
    float thermalGlow = localT * 0.5;
    vec3 emissionColor = palette(
      localT,
      vec3(0.15, 0.15, 0.25),
      vec3(0.4, 0.3, 0.2),
      vec3(1.0, 1.0, 1.0),
      vec3(0.0, 0.1, 0.2)
    );

    // ── ACCUMULATE COLOR ──
    // Scattering contribution
    vec3 scatterLight = sunColor * scatter * phase * transmittance;

    // Emission contribution (thermal glow proportional to T)
    vec3 emitLight = emissionColor * thermalGlow * mu_abs * MARCH_SIZE * transmittance;

    // Ambient density fog
    float fogDensity = density * 0.005;
    vec3 densityColor = palette(
      partState.x,  // Sk determines color
      vec3(0.05, 0.1, 0.2),
      vec3(0.3, 0.4, 0.5),
      vec3(1.0, 0.7, 0.4),
      vec3(0.0, 0.15, 0.25)
    );
    vec3 fogLight = densityColor * fogDensity * transmittance;

    color += scatterLight + emitLight + fogLight;
    totalDensity += density * MARCH_SIZE;

    // ── PHASE ACCUMULATION (harmonic resonator loop) ──
    float omega = 6.28318 * n_level * 1e3; // partition frequency
    float tau_p = 1.0 / (omega + 1.0);     // partition lag
    phaseAccum += omega * tau_p * MARCH_SIZE;

    // ── REFRACTION (RPI: scattering = information) ──
    if (uNetworkDensity > 0.3) {
      // Bend ray through density gradient (Snell's discrete law)
      float n_here = refractiveIndex(pos);
      float n_ahead = refractiveIndex(pos + rd * MARCH_SIZE * 2.0);
      if (abs(n_ahead - n_here) > 0.001) {
        vec3 gradN = vec3(
          refractiveIndex(pos + vec3(0.01, 0, 0)) - refractiveIndex(pos - vec3(0.01, 0, 0)),
          refractiveIndex(pos + vec3(0, 0.01, 0)) - refractiveIndex(pos - vec3(0, 0.01, 0)),
          refractiveIndex(pos + vec3(0, 0, 0.01)) - refractiveIndex(pos - vec3(0, 0, 0.01))
        );
        rd = normalize(rd + gradN * 0.02);
      }
    }

    // Early termination
    if (transmittance < 0.01) break;
  }

  // ── BACKGROUND ──
  vec3 bg = mix(
    vec3(0.02, 0.03, 0.06),  // dark bottom
    vec3(0.04, 0.06, 0.12),  // slightly lighter top
    uv.y * 0.5 + 0.5
  );
  color += transmittance * bg;

  // ── POST: subtle vignette ──
  vec2 uvCenter = (gl_FragCoord.xy / uResolution) - 0.5;
  float vignette = 1.0 - 0.3 * dot(uvCenter, uvCenter);
  color *= vignette;

  // Tonemap
  color = color / (color + 1.0);
  // Gamma
  color = pow(color, vec3(1.0 / 2.2));

  gl_FragColor = vec4(color, phaseAccum);
}
