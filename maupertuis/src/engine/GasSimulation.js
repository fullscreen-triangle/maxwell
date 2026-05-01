import * as THREE from 'three';
import { useMemo, useRef, useState } from 'react';
import { createPortal, useFrame, extend } from '@react-three/fiber';
import { useFBO } from '@react-three/drei';
import { perlinNoise3D, simplexNoise3D, curlNoise } from './shaders/noiseLib.glsl';

// ═══════════════════════════════════════════════════════════
// SimulationMaterial — runs in FBO, advects particle positions
// via curl noise (divergence-free) modulated by physics.
// Each pixel = 1 particle. RGBA = (x, y, z, life).
//
// Temperature → noise frequency (uCurlFreq)
// Network density → number of curl octaves (gas: 1, liquid: 4)
// Volume → bounding sphere radius
// ═══════════════════════════════════════════════════════════

class SimulationMaterial extends THREE.ShaderMaterial {
  constructor() {
    // Initial sphere distribution
    const size = 256;
    const data = new Float32Array(size * size * 4);
    const v = new THREE.Vector3();
    for (let i = 0; i < size * size; i++) {
      do {
        v.set(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
      } while (v.length() > 1);
      v.normalize().multiplyScalar(0.6 + Math.random() * 0.4);
      data[i * 4 + 0] = v.x;
      data[i * 4 + 1] = v.y;
      data[i * 4 + 2] = v.z;
      data[i * 4 + 3] = Math.random();
    }
    const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.FloatType);
    tex.needsUpdate = true;

    super({
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D positions;
        uniform float uTime;
        uniform float uCurlFreq;
        uniform float uNetworkDensity;
        uniform float uVolume;
        uniform float uTemperature;
        varying vec2 vUv;

        ${simplexNoise3D}
        ${perlinNoise3D}
        ${curlNoise}

        void main() {
          float t = uTime * 0.015 * (uTemperature / 300.0);
          vec3 base = texture2D(positions, vUv).rgb;
          vec3 pos = base;
          vec3 curlPos = base;

          // Single curl
          pos = curl(pos * uCurlFreq + t);

          // Multi-octave (more octaves in liquid phase)
          curlPos = curl(curlPos * uCurlFreq + t);
          curlPos += curl(curlPos * uCurlFreq * 2.0) * 0.5 * uNetworkDensity;
          curlPos += curl(curlPos * uCurlFreq * 4.0) * 0.25 * uNetworkDensity;
          curlPos += curl(curlPos * uCurlFreq * 8.0) * 0.125 * uNetworkDensity;
          curlPos += curl(pos * uCurlFreq * 16.0) * 0.0625 * uNetworkDensity;

          // Mix single vs multi based on Perlin
          vec3 finalPos = mix(pos, curlPos, cnoise(pos + t));

          // Confine to volume (gentle pull toward origin if outside)
          float r = length(finalPos);
          float maxR = uVolume * 1.5;
          if (r > maxR) {
            finalPos *= maxR / r;
          }

          gl_FragColor = vec4(finalPos, 1.0);
        }
      `,
      uniforms: {
        positions: { value: tex },
        uTime: { value: 0 },
        uCurlFreq: { value: 0.25 },
        uNetworkDensity: { value: 0.0 },
        uVolume: { value: 1.0 },
        uTemperature: { value: 300.0 },
      },
    });
  }
}

// ═══════════════════════════════════════════════════════════
// Render material — circular DOF points colored by physics
// Color encodes: temperature (warm/cool), network density (saturation)
// Size scales with thermal jitter
// ═══════════════════════════════════════════════════════════

class DofPointsMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      vertexShader: /* glsl */ `
        uniform sampler2D positions;
        uniform float uTime;
        uniform float uFocus;
        uniform float uFov;
        uniform float uBlur;
        uniform float uTemperature;
        varying float vDistance;
        varying float vRadius;
        void main() {
          vec3 pos = texture2D(positions, position.xy).xyz;
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          vDistance = abs(uFocus - -mvPosition.z);
          vRadius = length(pos);
          gl_PointSize = (step(1.0 - (1.0 / uFov), position.x)) * vDistance * uBlur * 2.0;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uOpacity;
        uniform float uTemperature;
        uniform float uNetworkDensity;
        varying float vDistance;
        varying float vRadius;

        vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
          return a + b * cos(6.28318 * (c * t + d));
        }

        void main() {
          vec2 cxy = 2.0 * gl_PointCoord - 1.0;
          float r2 = dot(cxy, cxy);
          if (r2 > 1.0) discard;

          // Cool blue (cold gas) → warm red (hot gas)
          float tempFactor = clamp(uTemperature / 1000.0, 0.0, 1.0);
          vec3 coldColor = vec3(0.5, 0.85, 1.0);   // light cyan
          vec3 hotColor  = vec3(1.0, 0.6, 0.4);    // warm coral
          vec3 baseColor = mix(coldColor, hotColor, tempFactor);

          // Liquid: tint toward primaryDark (cyan)
          vec3 liquidTint = vec3(0.34, 0.9, 0.85);
          baseColor = mix(baseColor, liquidTint, uNetworkDensity * 0.4);

          // Soft circular falloff
          float alpha = (1.0 - sqrt(r2)) * (1.04 - clamp(vDistance * 1.5, 0.0, 1.0));

          // Boost center brightness (additive sparkle)
          baseColor += vec3(0.3) * pow(1.0 - sqrt(r2), 4.0);

          gl_FragColor = vec4(baseColor, alpha * uOpacity);
        }
      `,
      uniforms: {
        positions: { value: null },
        uTime: { value: 0 },
        uFocus: { value: 5.1 },
        uFov: { value: 50 },
        uBlur: { value: 30 },
        uOpacity: { value: 0.9 },
        uTemperature: { value: 300 },
        uNetworkDensity: { value: 0.0 },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
  }
}

extend({ SimulationMaterial, DofPointsMaterial });

// ═══════════════════════════════════════════════════════════
// Main Particles component
// ═══════════════════════════════════════════════════════════

export default function GasSimulation({ params, onReadouts }) {
  const simRef = useRef();
  const renderRef = useRef();
  const SIZE = 256;

  const [scene] = useState(() => new THREE.Scene());
  const [camera] = useState(() =>
    new THREE.OrthographicCamera(-1, 1, 1, -1, 1 / Math.pow(2, 53), 1)
  );

  const target = useFBO(SIZE, SIZE, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat,
    type: THREE.FloatType,
  });

  // Quad for FBO render
  const quadPositions = useMemo(
    () => new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0]),
    []
  );
  const quadUvs = useMemo(
    () => new Float32Array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]),
    []
  );

  // Particle UVs (each particle's address into the position texture)
  const particles = useMemo(() => {
    const length = SIZE * SIZE;
    const arr = new Float32Array(length * 3);
    for (let i = 0; i < length; i++) {
      arr[i * 3 + 0] = (i % SIZE) / SIZE;
      arr[i * 3 + 1] = i / SIZE / SIZE;
      arr[i * 3 + 2] = 0;
    }
    return arr;
  }, []);

  const readoutTimer = useRef(0);
  const frameCount = useRef(0);

  useFrame((state, delta) => {
    // Render simulation into FBO
    state.gl.setRenderTarget(target);
    state.gl.clear();
    state.gl.render(scene, camera);
    state.gl.setRenderTarget(null);

    // Update render material
    if (renderRef.current) {
      renderRef.current.uniforms.positions.value = target.texture;
      renderRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      renderRef.current.uniforms.uTemperature.value = params.temperature || 300;
      renderRef.current.uniforms.uNetworkDensity.value = params.networkDensity || 0;

      // DOF parameters from physics
      const focus = 5.1;
      const fov = 50;
      const aperture = params.networkDensity > 0.3 ? 2.5 : 1.8; // tighter focus for liquid
      const targetBlur = (5.6 - aperture) * 9;
      renderRef.current.uniforms.uFocus.value = THREE.MathUtils.lerp(
        renderRef.current.uniforms.uFocus.value, focus, 0.1
      );
      renderRef.current.uniforms.uFov.value = THREE.MathUtils.lerp(
        renderRef.current.uniforms.uFov.value, fov, 0.1
      );
      renderRef.current.uniforms.uBlur.value = THREE.MathUtils.lerp(
        renderRef.current.uniforms.uBlur.value, targetBlur, 0.1
      );
    }

    // Update simulation material
    if (simRef.current) {
      const speed = (params.temperature || 300) / 300; // hotter = faster
      simRef.current.uniforms.uTime.value = state.clock.elapsedTime * 100 * speed;
      simRef.current.uniforms.uTemperature.value = params.temperature || 300;
      simRef.current.uniforms.uNetworkDensity.value = params.networkDensity || 0;
      simRef.current.uniforms.uVolume.value = params.volume || 1.0;

      // Curl frequency from network density: gas = low (open swirls), liquid = high (tight eddies)
      const targetCurl = 0.15 + (params.networkDensity || 0) * 0.3;
      simRef.current.uniforms.uCurlFreq.value = THREE.MathUtils.lerp(
        simRef.current.uniforms.uCurlFreq.value, targetCurl, 0.05
      );
    }

    frameCount.current++;

    // Diagnostics
    readoutTimer.current += delta;
    if (onReadouts && readoutTimer.current > 0.2) {
      readoutTimer.current = 0;
      const kB = 1.380649e-23;
      const N = params.particles || 200;
      const T = params.temperature || 300;
      const V_m3 = (params.volume || 1.0) * 1e-24;
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
    <>
      {/* Simulation pass into FBO */}
      {createPortal(
        <mesh>
          <simulationMaterial ref={simRef} />
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={quadPositions.length / 3}
              array={quadPositions}
              itemSize={3}
            />
            <bufferAttribute
              attach="attributes-uv"
              count={quadUvs.length / 2}
              array={quadUvs}
              itemSize={2}
            />
          </bufferGeometry>
        </mesh>,
        scene
      )}

      {/* Render the points */}
      <points>
        <dofPointsMaterial ref={renderRef} />
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={particles.length / 3}
            array={particles}
            itemSize={3}
          />
        </bufferGeometry>
      </points>
    </>
  );
}
