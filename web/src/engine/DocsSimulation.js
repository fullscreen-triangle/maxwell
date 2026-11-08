import * as THREE from 'three';
import { useMemo, useRef, useState } from 'react';
import { createPortal, useFrame, extend } from '@react-three/fiber';
import { useFBO } from '@react-three/drei';
import { perlinNoise3D, simplexNoise3D, curlNoise } from './shaders/noiseLib.glsl';

// Compact GPGPU particle system designed for docs sidebar.
// 128×128 = 16,384 particles, smooth interpolation between scenes.

class DocsSimMaterial extends THREE.ShaderMaterial {
  constructor() {
    const size = 128;
    const data = new Float32Array(size * size * 4);
    const v = new THREE.Vector3();
    for (let i = 0; i < size * size; i++) {
      do {
        v.set(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
      } while (v.length() > 1);
      v.normalize().multiplyScalar(0.4 + Math.random() * 0.6);
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
        uniform float uOctaves;
        uniform float uTurbulence;
        uniform vec3 uFlowDir;
        varying vec2 vUv;

        ${simplexNoise3D}
        ${perlinNoise3D}
        ${curlNoise}

        void main() {
          float t = uTime * 0.015 * (uTemperature / 300.0);
          vec3 base = texture2D(positions, vUv).rgb;
          vec3 pos = base;
          vec3 curlPos = base;

          pos = curl(pos * uCurlFreq + t);
          curlPos = curl(curlPos * uCurlFreq + t);

          if (uOctaves > 0.5) {
            curlPos += curl(curlPos * uCurlFreq * 2.0) * 0.5 * uNetworkDensity;
          }
          if (uOctaves > 1.5) {
            curlPos += curl(curlPos * uCurlFreq * 4.0) * 0.25 * uNetworkDensity;
          }
          if (uOctaves > 2.5) {
            curlPos += curl(curlPos * uCurlFreq * 8.0) * 0.125 * uNetworkDensity;
          }
          if (uOctaves > 3.5) {
            curlPos += curl(pos * uCurlFreq * 16.0) * 0.0625 * uTurbulence;
          }

          vec3 finalPos = mix(pos, curlPos, cnoise(pos + t));

          // Add directional bias (Poiseuille flow, etc.)
          finalPos += uFlowDir * 0.1;

          // Confine to volume
          float r = length(finalPos);
          float maxR = uVolume * 1.4;
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
        uOctaves: { value: 1.0 },
        uTurbulence: { value: 0.0 },
        uFlowDir: { value: new THREE.Vector3() },
      },
    });
  }
}

class DocsPointsMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      vertexShader: /* glsl */ `
        uniform sampler2D positions;
        uniform float uFocus;
        uniform float uFov;
        uniform float uBlur;
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
        uniform vec3 uColorA;
        uniform vec3 uColorB;
        varying float vDistance;
        varying float vRadius;

        void main() {
          vec2 cxy = 2.0 * gl_PointCoord - 1.0;
          float r2 = dot(cxy, cxy);
          if (r2 > 1.0) discard;

          float tempFactor = clamp(uTemperature / 1000.0, 0.0, 1.0);
          vec3 baseColor = mix(uColorA, uColorB, tempFactor);

          // Liquid tint
          baseColor = mix(baseColor, vec3(0.34, 0.9, 0.85), uNetworkDensity * 0.4);

          float alpha = (1.0 - sqrt(r2)) * (1.04 - clamp(vDistance * 1.5, 0.0, 1.0));
          baseColor += vec3(0.3) * pow(1.0 - sqrt(r2), 4.0);

          gl_FragColor = vec4(baseColor, alpha * uOpacity);
        }
      `,
      uniforms: {
        positions: { value: null },
        uFocus: { value: 5.1 },
        uFov: { value: 50 },
        uBlur: { value: 30 },
        uOpacity: { value: 0.85 },
        uTemperature: { value: 300 },
        uNetworkDensity: { value: 0.0 },
        uColorA: { value: new THREE.Color('#7fc8e8') },
        uColorB: { value: new THREE.Color('#ff9f6b') },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
  }
}

extend({ DocsSimMaterial, DocsPointsMaterial });

const DEFAULT_PARAMS = {
  temperature: 300,
  networkDensity: 0.0,
  volume: 1.0,
  curlFreq: 0.2,
  octaves: 1,
  turbulence: 0.0,
  flowDir: [0, 0, 0],
  colorA: '#7fc8e8',
  colorB: '#ff9f6b',
};

export default function DocsSimulation({ params = DEFAULT_PARAMS }) {
  const simRef = useRef();
  const renderRef = useRef();
  const SIZE = 128;

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

  const quadPositions = useMemo(
    () => new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0]),
    []
  );
  const quadUvs = useMemo(
    () => new Float32Array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]),
    []
  );

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

  // Smoothed/lerped uniform values for nice transitions between sections
  const stateRef = useRef({
    temperature: params.temperature || 300,
    networkDensity: params.networkDensity || 0,
    volume: params.volume || 1,
    curlFreq: params.curlFreq || 0.2,
    octaves: params.octaves || 1,
    turbulence: params.turbulence || 0,
    flowDir: new THREE.Vector3(...(params.flowDir || [0, 0, 0])),
    colorA: new THREE.Color(params.colorA || '#7fc8e8'),
    colorB: new THREE.Color(params.colorB || '#ff9f6b'),
  });

  useFrame((state, delta) => {
    state.gl.setRenderTarget(target);
    state.gl.clear();
    state.gl.render(scene, camera);
    state.gl.setRenderTarget(null);

    // Smooth transitions
    const lerp = THREE.MathUtils.lerp;
    const k = 0.04; // smoothing rate

    const s = stateRef.current;
    s.temperature = lerp(s.temperature, params.temperature ?? 300, k);
    s.networkDensity = lerp(s.networkDensity, params.networkDensity ?? 0, k);
    s.volume = lerp(s.volume, params.volume ?? 1, k);
    s.curlFreq = lerp(s.curlFreq, params.curlFreq ?? 0.2, k);
    s.octaves = lerp(s.octaves, params.octaves ?? 1, k);
    s.turbulence = lerp(s.turbulence, params.turbulence ?? 0, k);

    const targetFlow = params.flowDir || [0, 0, 0];
    s.flowDir.x = lerp(s.flowDir.x, targetFlow[0], k);
    s.flowDir.y = lerp(s.flowDir.y, targetFlow[1], k);
    s.flowDir.z = lerp(s.flowDir.z, targetFlow[2], k);

    const tcA = new THREE.Color(params.colorA || '#7fc8e8');
    const tcB = new THREE.Color(params.colorB || '#ff9f6b');
    s.colorA.lerp(tcA, k);
    s.colorB.lerp(tcB, k);

    // Apply to materials
    if (simRef.current) {
      const speed = s.temperature / 300;
      simRef.current.uniforms.uTime.value = state.clock.elapsedTime * 100 * speed;
      simRef.current.uniforms.uTemperature.value = s.temperature;
      simRef.current.uniforms.uNetworkDensity.value = s.networkDensity;
      simRef.current.uniforms.uVolume.value = s.volume;
      simRef.current.uniforms.uCurlFreq.value = s.curlFreq;
      simRef.current.uniforms.uOctaves.value = s.octaves;
      simRef.current.uniforms.uTurbulence.value = s.turbulence;
      simRef.current.uniforms.uFlowDir.value.copy(s.flowDir);
    }

    if (renderRef.current) {
      renderRef.current.uniforms.positions.value = target.texture;
      renderRef.current.uniforms.uTemperature.value = s.temperature;
      renderRef.current.uniforms.uNetworkDensity.value = s.networkDensity;
      renderRef.current.uniforms.uColorA.value.copy(s.colorA);
      renderRef.current.uniforms.uColorB.value.copy(s.colorB);
    }
  });

  return (
    <>
      {createPortal(
        <mesh>
          <docsSimMaterial ref={simRef} />
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

      <points>
        <docsPointsMaterial ref={renderRef} />
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
