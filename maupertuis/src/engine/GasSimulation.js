import { useRef, useMemo, useCallback } from 'react';
import { useFrame, useThree, createPortal } from '@react-three/fiber';
import * as THREE from 'three';

// ═══════════════════════════════════════════════════════════
// Standalone volumetric ray march — no FBO complexity.
// Renders a fullscreen fragment shader that procedurally
// generates the gas/fluid volume from uniforms alone.
// The "particles" are implicit: noise-seeded density fields
// modulated by S-entropy parameters.
// ═══════════════════════════════════════════════════════════

const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const fragmentShader = /* glsl */ `
precision highp float;

uniform float uTime;
uniform vec2 uResolution;
uniform float uTemperature;
uniform float uVolume;
uniform float uParticleCount;
uniform float uNetworkDensity;
uniform vec3 uCameraPos;
uniform vec3 uCameraTarget;

#define PI 3.14159265359
#define MAX_STEPS 80
#define MARCH_SIZE 0.0125
#define ABSORPTION 1.2
#define SCATTER_G 0.3

// ─── Hash / Noise ─────────────────────────────────────────
float hash(vec3 p) {
  p = fract(p * vec3(443.897, 441.423, 437.195));
  p += dot(p, p.yzx + 19.19);
  return fract((p.x + p.y) * p.z);
}

float noise3D(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec3(1,0,0));
  float c = hash(i + vec3(0,1,0));
  float d = hash(i + vec3(1,1,0));
  float e = hash(i + vec3(0,0,1));
  float f1 = hash(i + vec3(1,0,1));
  float g = hash(i + vec3(0,1,1));
  float h1 = hash(i + vec3(1,1,1));
  return mix(mix(mix(a,b,f.x), mix(c,d,f.x), f.y),
             mix(mix(e,f1,f.x), mix(g,h1,f.x), f.y), f.z);
}

float fbm(vec3 p, int octaves) {
  float val = 0.0;
  float amp = 0.5;
  float freq = 1.0;
  for (int i = 0; i < 6; i++) {
    if (i >= octaves) break;
    val += amp * noise3D(p * freq);
    freq *= 2.03;
    amp *= 0.48;
  }
  return val;
}

// ─── Particle density field ───────────────────────────────
// Generates N "particles" as noise-seeded blobs in [0, vol]^3
float particleDensity(vec3 pos, float time) {
  float vol = uVolume;
  // Normalize position to [0,1]
  vec3 np = pos / vol;
  if (any(lessThan(np, vec3(-0.05))) || any(greaterThan(np, vec3(1.05)))) return 0.0;

  float density = 0.0;

  // Thermal motion: higher T = faster movement
  float thermalSpeed = sqrt(uTemperature / 300.0) * 0.3;

  // Macro-scale density from FBM (represents statistical particle distribution)
  vec3 moving = np * 4.0 + vec3(time * thermalSpeed * 0.7, time * thermalSpeed * 0.5, time * thermalSpeed * 0.3);
  float macro = fbm(moving, 5);

  // Individual particle blobs (discrete lumps modulated by count)
  float blobScale = 3.0 + uParticleCount * 0.02;
  vec3 blobPos = np * blobScale + vec3(time * thermalSpeed);
  float blobs = 0.0;
  for (int i = 0; i < 8; i++) {
    vec3 offset = vec3(
      hash(vec3(float(i) * 13.7, 0.0, 0.0)),
      hash(vec3(0.0, float(i) * 17.3, 0.0)),
      hash(vec3(0.0, 0.0, float(i) * 23.1))
    ) * 2.0 - 1.0;
    vec3 center = vec3(0.5) + offset * 0.35;
    // Animate: orbit + thermal jitter
    center += 0.15 * vec3(
      sin(time * thermalSpeed * (1.0 + float(i) * 0.3) + float(i)),
      cos(time * thermalSpeed * (0.8 + float(i) * 0.2) + float(i) * 2.0),
      sin(time * thermalSpeed * (0.6 + float(i) * 0.4) + float(i) * 3.0)
    );
    float d = length(np - center);
    float r = 0.06 + uNetworkDensity * 0.08; // wider blobs in liquid
    blobs += exp(-d * d / (2.0 * r * r));
  }

  // Network density controls how connected the medium is
  // Gas: isolated blobs. Liquid: smooth connected field
  float connection = smoothstep(0.0, 1.0, uNetworkDensity);
  density = mix(blobs * 0.5, macro * 2.0 + blobs * 0.3, connection);

  // Scale by particle count (more particles = denser)
  density *= (uParticleCount / 200.0);

  return max(density, 0.0);
}

// ─── Velocity field (curl-noise style) ────────────────────
vec3 velocityField(vec3 pos, float time) {
  float thermalSpeed = sqrt(uTemperature / 300.0) * 0.5;
  vec3 np = pos / uVolume;
  float eps = 0.01;
  // Curl of noise = divergence-free velocity field
  float nx = noise3D((np + vec3(eps,0,0)) * 3.0 + time * thermalSpeed * 0.4);
  float ny = noise3D((np + vec3(0,eps,0)) * 3.0 + time * thermalSpeed * 0.4);
  float nz = noise3D((np + vec3(0,0,eps)) * 3.0 + time * thermalSpeed * 0.4);
  float n0 = noise3D(np * 3.0 + time * thermalSpeed * 0.4);
  return vec3(ny - n0, nz - n0, nx - n0) * thermalSpeed;
}

// ─── Henyey-Greenstein ────────────────────────────────────
float HG(float g, float cosTheta) {
  float gg = g * g;
  return (1.0 / (4.0 * PI)) * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * cosTheta, 1.5));
}

// ─── Cosine palette ───────────────────────────────────────
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  return a + b * cos(6.28318 * (c * t + d));
}

// ─── Camera ───────────────────────────────────────────────
mat3 lookAt(vec3 eye, vec3 target) {
  vec3 f = normalize(target - eye);
  vec3 r = normalize(cross(vec3(0,1,0), f));
  vec3 u = cross(f, r);
  return mat3(r, u, f);
}

// ─── Main ─────────────────────────────────────────────────
void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / min(uResolution.x, uResolution.y);

  // Camera
  mat3 cam = lookAt(uCameraPos, uCameraTarget);
  vec3 rd = cam * normalize(vec3(uv, 1.2));
  vec3 ro = uCameraPos;

  // Sun
  vec3 sunDir = normalize(vec3(0.6, 0.8, 0.4));
  vec3 sunCol = vec3(1.0, 0.92, 0.8) * 2.0;

  // Ray march
  vec3 color = vec3(0.0);
  float transmittance = 1.0;
  float totalPhase = 0.0;

  // Stochastic offset (anti-banding)
  float t0 = hash(vec3(gl_FragCoord.xy, uTime * 17.31)) * MARCH_SIZE;

  for (int i = 0; i < MAX_STEPS; i++) {
    float t = t0 + float(i) * MARCH_SIZE;
    vec3 pos = ro + rd * t;

    // Volume bounds check
    if (any(lessThan(pos, vec3(-0.1))) || any(greaterThan(pos, vec3(uVolume + 0.1)))) {
      if (t > 3.0) break;
      continue;
    }

    float density = particleDensity(pos, uTime);
    if (density < 0.01) continue;

    // ── Partition state from position ──
    vec3 np = pos / uVolume;
    float Sk = fbm(np * 2.0 + uTime * 0.1, 3); // knowledge entropy
    float nLevel = floor(Sk * 4.99) + 1.0;

    // ── ABSORPTION (Beer-Lambert) ──
    float mu_abs = ABSORPTION * density * 0.15 * (nLevel / 5.0);
    float stepTrans = exp(-mu_abs * MARCH_SIZE);
    float absorbed = (1.0 - stepTrans);

    // ── EMISSION (thermal glow) ──
    float tempFactor = uTemperature / 300.0;
    vec3 emitColor = palette(
      nLevel / 5.0 * tempFactor,
      vec3(0.1, 0.12, 0.2),
      vec3(0.4, 0.25, 0.15),
      vec3(1.0, 0.8, 0.6),
      vec3(0.0, 0.1, 0.2)
    );
    // Hotter = brighter emission
    vec3 emission = emitColor * absorbed * tempFactor * 0.8;

    // ── SCATTERING ──
    float cosTheta = dot(rd, sunDir);
    float scatterStrength;
    if (uNetworkDensity < 0.3) {
      // Gas: Rayleigh
      scatterStrength = 0.04 * pow(nLevel / 5.0, 4.0);
    } else {
      // Liquid: Mie (broader, brighter)
      scatterStrength = 0.15 * pow(nLevel / 5.0, 2.0);
    }
    float phase = HG(SCATTER_G, cosTheta);
    vec3 scatterLight = sunCol * scatterStrength * phase * density * MARCH_SIZE;

    // ── REFRACTION (liquid mode) ──
    if (uNetworkDensity > 0.3) {
      // Density gradient bends ray
      float eps = MARCH_SIZE;
      float dR = particleDensity(pos + vec3(eps,0,0), uTime) - particleDensity(pos - vec3(eps,0,0), uTime);
      float dU = particleDensity(pos + vec3(0,eps,0), uTime) - particleDensity(pos - vec3(0,eps,0), uTime);
      float dF = particleDensity(pos + vec3(0,0,eps), uTime) - particleDensity(pos - vec3(0,0,eps), uTime);
      vec3 gradDensity = vec3(dR, dU, dF) / (2.0 * eps);
      rd = normalize(rd + gradDensity * 0.0004 * uNetworkDensity);
    }

    // ── VELOCITY VISUALIZATION ──
    vec3 vel = velocityField(pos, uTime);
    float velMag = length(vel);
    vec3 velColor = palette(
      velMag * 2.0,
      vec3(0.02, 0.05, 0.15),
      vec3(0.15, 0.2, 0.3),
      vec3(0.8, 0.6, 0.4),
      vec3(0.25, 0.15, 0.05)
    );

    // ── ACCUMULATE ──
    color += transmittance * (emission + scatterLight + velColor * density * 0.02 * MARCH_SIZE);
    transmittance *= stepTrans;

    // Phase accumulation
    float omega = 6.28318 * nLevel;
    totalPhase += omega * MARCH_SIZE;

    if (transmittance < 0.005) break;
  }

  // ── Background gradient ──
  vec3 bg = mix(
    vec3(0.01, 0.015, 0.04),
    vec3(0.03, 0.04, 0.08),
    uv.y + 0.5
  );
  color += transmittance * bg;

  // ── Container wireframe ──
  float vol = uVolume;
  for (int i = 0; i < MAX_STEPS; i++) {
    float t = float(i) * MARCH_SIZE;
    vec3 p = ro + rd * t;
    vec3 np = p;
    // Edge glow at container boundaries
    float edgeX = min(abs(np.x), abs(np.x - vol));
    float edgeY = min(abs(np.y), abs(np.y - vol));
    float edgeZ = min(abs(np.z), abs(np.z - vol));
    float edge = min(edgeX, min(edgeY, edgeZ));
    // Only show edge where 2 of 3 coords are near boundary
    float near1 = step(edgeX, 0.008) + step(edgeY, 0.008) + step(edgeZ, 0.008);
    if (near1 >= 2.0 && edge < 0.008) {
      float glow = smoothstep(0.008, 0.0, edge) * 0.25;
      color += vec3(0.15, 0.3, 0.4) * glow;
      break;
    }
    if (t > 3.0) break;
  }

  // ── Post-processing ──
  // Vignette
  vec2 vigUv = gl_FragCoord.xy / uResolution - 0.5;
  color *= 1.0 - 0.4 * dot(vigUv, vigUv);

  // Tonemap (ACES-ish)
  color = color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14);

  // Gamma
  color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));

  gl_FragColor = vec4(color, 1.0);
}
`;

export default function GasSimulation({ params, onReadouts }) {
  const meshRef = useRef();
  const { size } = useThree();

  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(size.width, size.height) },
        uTemperature: { value: 300 },
        uVolume: { value: 0.8 },
        uParticleCount: { value: 200 },
        uNetworkDensity: { value: 0.0 },
        uCameraPos: { value: new THREE.Vector3(0.4, 0.5, 1.8) },
        uCameraTarget: { value: new THREE.Vector3(0.4, 0.35, 0.0) },
      },
      depthTest: false,
      depthWrite: false,
    });
  }, []);

  // Update resolution on resize
  useMemo(() => {
    material.uniforms.uResolution.value.set(size.width, size.height);
  }, [size, material]);

  const readoutTimer = useRef(0);
  const frameCount = useRef(0);

  useFrame((state, delta) => {
    material.uniforms.uTime.value = state.clock.elapsedTime;
    material.uniforms.uTemperature.value = params.temperature || 300;
    material.uniforms.uVolume.value = params.volume || 0.8;
    material.uniforms.uParticleCount.value = params.particles || 200;
    material.uniforms.uNetworkDensity.value = params.networkDensity || 0.0;
    material.uniforms.uCameraPos.value.copy(state.camera.position);

    // Camera target from orbit controls
    const target = new THREE.Vector3(0.4, 0.35, 0.4);
    material.uniforms.uCameraTarget.value.copy(target);

    frameCount.current++;

    // Diagnostics readback
    readoutTimer.current += delta;
    if (onReadouts && readoutTimer.current > 0.2) {
      readoutTimer.current = 0;
      const kB = 1.380649e-23;
      const N = params.particles || 200;
      const T = params.temperature || 300;
      const V_m3 = (params.volume || 0.8) * 1e-24;
      const P = N * kB * T / V_m3;
      const U = 1.5 * N * kB * T;
      const nd = params.networkDensity || 0.0;
      const mu = nd > 0.3
        ? (params.tau_c || 0.15e-12) * (params.g_coupling || 6.6) * 1e9
        : 0;
      onReadouts({
        T: T.toFixed(1),
        P: P.toExponential(2),
        U: U.toExponential(3),
        S: '—',
        N: N,
        mu: mu > 0 ? mu.toFixed(2) + ' mPa·s' : '—',
        fps: Math.round(1 / Math.max(delta, 0.001)),
        frame: frameCount.current,
        rhoC: nd.toFixed(2),
      });
    }
  });

  return (
    <mesh ref={meshRef} frustumCulled={false} material={material}>
      <planeGeometry args={[2, 2]} />
    </mesh>
  );
}
