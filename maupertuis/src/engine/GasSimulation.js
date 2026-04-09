import { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { SimulationMaterial } from './SimulationMaterial';
import rayMarchFrag from './shaders/rayMarch.glsl';

const fullscreenVert = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// Initialize particle positions as DataTexture
function createPositionTexture(count) {
  const size = Math.ceil(Math.sqrt(count));
  const data = new Float32Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    data[i * 4 + 0] = Math.random() * 0.8 + 0.1; // Sk
    data[i * 4 + 1] = Math.random() * 0.8 + 0.1; // St
    data[i * 4 + 2] = Math.random() * 0.8 + 0.1; // Se
    data[i * 4 + 3] = Math.random() * 0.01;       // speed
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  return tex;
}

function createVelocityTexture(count, temperature) {
  const size = Math.ceil(Math.sqrt(count));
  const data = new Float32Array(size * size * 4);
  const sigma = Math.sqrt(temperature / 300.0) * 0.003;
  for (let i = 0; i < size * size; i++) {
    // Box-Muller for Gaussian velocities
    const u1 = Math.max(Math.random(), 1e-6);
    const u2 = Math.random();
    const u3 = Math.random();
    const u4 = Math.max(Math.random(), 1e-6);
    data[i * 4 + 0] = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    data[i * 4 + 1] = sigma * Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
    data[i * 4 + 2] = sigma * Math.sqrt(-2 * Math.log(u4)) * Math.cos(2 * Math.PI * u3);
    data[i * 4 + 3] = 1.0; // n_level
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  return tex;
}

export default function GasSimulation({ params, onReadouts }) {
  const { gl, size, camera } = useThree();
  const simMeshRef = useRef();
  const renderMeshRef = useRef();

  const count = params.particles || 200;
  const fboSize = Math.ceil(Math.sqrt(count));

  // FBO ping-pong for simulation
  const [fboA, fboB] = useMemo(() => {
    const opts = {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.FloatType,
    };
    return [
      new THREE.WebGLRenderTarget(fboSize, fboSize, opts),
      new THREE.WebGLRenderTarget(fboSize, fboSize, opts),
    ];
  }, [fboSize]);

  const velocityFBO = useMemo(() => {
    return new THREE.WebGLRenderTarget(fboSize, fboSize, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.FloatType,
    });
  }, [fboSize]);

  // Initialize textures
  const initPositions = useMemo(() => createPositionTexture(count), [count]);
  const initVelocities = useMemo(
    () => createVelocityTexture(count, params.temperature || 300),
    [count, params.temperature]
  );

  // Simulation material (GPGPU pass)
  const simMaterial = useMemo(() => new SimulationMaterial(), []);

  // Ray march material (render pass)
  const renderMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: fullscreenVert,
      fragmentShader: rayMarchFrag,
      uniforms: {
        uPositions: { value: null },
        uVelocities: { value: null },
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(size.width, size.height) },
        uTemperature: { value: params.temperature || 300 },
        uVolume: { value: params.volume || 1.0 },
        uParticleCount: { value: count },
        uNetworkDensity: { value: params.networkDensity || 0.0 },
        uCameraPos: { value: new THREE.Vector3(0.5, 0.5, 2.0) },
        uCameraMatrix: { value: new THREE.Matrix4() },
      },
    });
  }, []);

  // Scene + camera for FBO rendering
  const simScene = useMemo(() => new THREE.Scene(), []);
  const simCamera = useMemo(() => new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), []);

  // Sim quad
  useEffect(() => {
    const geom = new THREE.PlaneGeometry(2, 2);
    const mesh = new THREE.Mesh(geom, simMaterial);
    simScene.add(mesh);
    simMeshRef.current = mesh;

    // Seed initial positions into fboA
    const initScene = new THREE.Scene();
    const initMat = new THREE.MeshBasicMaterial({ map: initPositions });
    initScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), initMat));
    gl.setRenderTarget(fboA);
    gl.render(initScene, simCamera);

    const initVelScene = new THREE.Scene();
    const initVelMat = new THREE.MeshBasicMaterial({ map: initVelocities });
    initVelScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), initVelMat));
    gl.setRenderTarget(velocityFBO);
    gl.render(initVelScene, simCamera);

    gl.setRenderTarget(null);
    initScene.children.forEach((c) => { c.geometry.dispose(); c.material.dispose(); });
    initVelScene.children.forEach((c) => { c.geometry.dispose(); c.material.dispose(); });

    return () => {
      geom.dispose();
      simMaterial.dispose();
    };
  }, []);

  // Frame counter for ping-pong
  const frameRef = useRef(0);
  const readoutTimer = useRef(0);

  useFrame((state, delta) => {
    const frame = frameRef.current;
    const readFBO = frame % 2 === 0 ? fboA : fboB;
    const writeFBO = frame % 2 === 0 ? fboB : fboA;

    // ── PASS 0-2: Simulation (GPGPU) ──
    simMaterial.uniforms.positions.value = readFBO.texture;
    simMaterial.uniforms.velocities.value = velocityFBO.texture;
    simMaterial.uniforms.uTime.value = state.clock.elapsedTime;
    simMaterial.uniforms.uDeltaTime.value = Math.min(delta, 0.05);
    simMaterial.uniforms.uTemperature.value = params.temperature || 300;
    simMaterial.uniforms.uVolume.value = params.volume || 1.0;
    simMaterial.uniforms.uParticleCount.value = count;
    simMaterial.uniforms.uNetworkDensity.value = params.networkDensity || 0.0;
    simMaterial.uniforms.uResolution.value.set(fboSize, fboSize);

    gl.setRenderTarget(writeFBO);
    gl.render(simScene, simCamera);
    gl.setRenderTarget(null);

    // ── PASS 3: Ray march (fullscreen) ──
    renderMaterial.uniforms.uPositions.value = writeFBO.texture;
    renderMaterial.uniforms.uVelocities.value = velocityFBO.texture;
    renderMaterial.uniforms.uTime.value = state.clock.elapsedTime;
    renderMaterial.uniforms.uResolution.value.set(size.width, size.height);
    renderMaterial.uniforms.uTemperature.value = params.temperature || 300;
    renderMaterial.uniforms.uVolume.value = params.volume || 1.0;
    renderMaterial.uniforms.uParticleCount.value = count;
    renderMaterial.uniforms.uNetworkDensity.value = params.networkDensity || 0.0;

    // Camera
    renderMaterial.uniforms.uCameraPos.value.copy(state.camera.position);
    const camMat = new THREE.Matrix4();
    camMat.extractRotation(state.camera.matrixWorld);
    renderMaterial.uniforms.uCameraMatrix.value.copy(camMat);

    frameRef.current++;

    // ── PASS 4: Diagnostics (CPU readback, throttled) ──
    readoutTimer.current += delta;
    if (onReadouts && readoutTimer.current > 0.25) {
      readoutTimer.current = 0;
      const kB = 1.380649e-23;
      const N = count;
      const T = params.temperature || 300;
      const V = params.volume || 1.0;
      const P = N * kB * T / (V * 1e-24); // scaled
      const U = 1.5 * N * kB * T;
      const networkDens = params.networkDensity || 0.0;
      const mu = networkDens > 0.3
        ? (params.tau_c || 0.15e-12) * (params.g_coupling || 6.6) * 1e9
        : 0;
      onReadouts({
        T: T.toFixed(1),
        P: (P * 1e-21).toExponential(2),
        U: U.toExponential(3),
        S: (N * kB * (Math.log(V / N * Math.pow(4 * Math.PI * U / (3 * N * 6.626e-34 * 6.626e-34), 1.5)) + 2.5)).toExponential(3),
        N: N,
        mu: mu > 0 ? mu.toFixed(2) + ' mPa·s' : '—',
        fps: (1 / delta).toFixed(0),
        frame: frameRef.current,
        rhoC: (networkDens).toFixed(2),
      });
    }
  });

  return (
    <mesh ref={renderMeshRef} material={renderMaterial}>
      <planeGeometry args={[2, 2]} />
    </mesh>
  );
}
