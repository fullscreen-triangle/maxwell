import { useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import dynamic from 'next/dynamic';
import Head from 'next/head';
import { PRESETS } from '@/lib/presets';

const GasSimulation = dynamic(() => import('@/engine/GasSimulation'), { ssr: false });

function Slider({ label, value, min, max, step, onChange, unit }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="text-primaryDark font-mono">
          {typeof value === 'number' ? value.toFixed(step < 1 ? 2 : 0) : value}
          {unit && <span className="text-gray-500 ml-1">{unit}</span>}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                   [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-primaryDark
                   [&::-webkit-slider-thumb]:rounded-full"
      />
    </div>
  );
}

function ReadoutRow({ label, value }) {
  return (
    <div className="flex justify-between text-xs font-mono py-0.5">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-200">{value}</span>
    </div>
  );
}

export default function SimulatePage() {
  const [params, setParams] = useState({
    particles: 200,
    temperature: 300,
    volume: 0.8,
    networkDensity: 0.0,
    mode: 'gas',
  });

  const [readouts, setReadouts] = useState({
    T: '300.0', P: '—', U: '—', S: '—', N: 200,
    mu: '—', fps: '—', frame: 0, rhoC: '0.00',
  });

  const [preset, setPreset] = useState('ideal-gas-300k');
  const [playing, setPlaying] = useState(true);

  const handlePreset = useCallback((key) => {
    const p = PRESETS[key];
    if (!p) return;
    setPreset(key);
    setParams({
      particles: p.particles,
      temperature: p.temperature,
      volume: p.volume || 0.8,
      networkDensity: p.mode === 'fluid' ? 0.8 : 0.0,
      mode: p.mode,
      tau_c: p.tau_c,
      g_coupling: p.g_coupling,
    });
  }, []);

  const updateParam = useCallback((key, val) => {
    setParams((prev) => ({ ...prev, [key]: val }));
  }, []);

  return (
    <>
      <Head>
        <title>Maupertuis — Simulate</title>
      </Head>

      <div className="fixed inset-0 bg-dark flex flex-col overflow-hidden">
        {/* ── Toolbar ── */}
        <div className="h-10 bg-dark/90 border-b border-gray-800 flex items-center px-4 gap-4 z-10">
          <span className="text-primaryDark font-bold text-sm tracking-wider">MAUPERTUIS</span>

          <div className="flex gap-1 bg-gray-800 rounded-md p-0.5">
            <button
              onClick={() => updateParam('networkDensity', 0.0) || updateParam('mode', 'gas')}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                params.mode === 'gas' ? 'bg-primaryDark text-dark font-bold' : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Gas
            </button>
            <button
              onClick={() => updateParam('networkDensity', 0.8) || updateParam('mode', 'fluid')}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                params.mode === 'fluid' ? 'bg-primaryDark text-dark font-bold' : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              Fluid
            </button>
          </div>

          <select
            value={preset}
            onChange={(e) => handlePreset(e.target.value)}
            className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded border border-gray-700"
          >
            {Object.entries(PRESETS).map(([key, p]) => (
              <option key={key} value={key}>{p.name}</option>
            ))}
          </select>

          <div className="flex-1" />

          <span className="text-gray-600 text-xs">
            {readouts.fps} FPS &middot; Frame {readouts.frame}
          </span>
        </div>

        {/* ── Main area ── */}
        <div className="flex flex-1 overflow-hidden">
          {/* ── Canvas ── */}
          <div className="flex-1 relative">
            <Canvas
              camera={{ position: [0.5, 0.5, 2.0], fov: 50 }}
              gl={{ antialias: false, alpha: false, powerPreference: 'high-performance' }}
              dpr={0.75}
              frameloop={playing ? 'always' : 'demand'}
            >
              <color attach="background" args={['#0a0a0f']} />
              <GasSimulation params={params} onReadouts={setReadouts} />
              <OrbitControls
                target={[0.4, 0.4, 0.4]}
                enableDamping
                dampingFactor={0.05}
                minDistance={0.5}
                maxDistance={5}
              />
            </Canvas>
          </div>

          {/* ── Side Panel ── */}
          <div className="w-72 bg-dark/95 border-l border-gray-800 overflow-y-auto p-4 flex flex-col gap-5">
            {/* Controls */}
            <div>
              <h3 className="text-gray-400 text-xs font-bold uppercase tracking-wider mb-3">Controls</h3>
              <div className="flex flex-col gap-3">
                <Slider label="Temperature" value={params.temperature} min={50} max={2000}
                  step={10} onChange={(v) => updateParam('temperature', v)} unit="K" />
                <Slider label="Volume" value={params.volume} min={0.1} max={1.0}
                  step={0.01} onChange={(v) => updateParam('volume', v)} />
                <Slider label="Particles" value={params.particles} min={10} max={2000}
                  step={10} onChange={(v) => updateParam('particles', v)} />
                <Slider label="Network Density" value={params.networkDensity} min={0} max={1}
                  step={0.01} onChange={(v) => updateParam('networkDensity', v)} unit="ρ_C" />
              </div>
            </div>

            {/* Readouts */}
            <div>
              <h3 className="text-gray-400 text-xs font-bold uppercase tracking-wider mb-3">Thermodynamics</h3>
              <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-800">
                <ReadoutRow label="T" value={`${readouts.T} K`} />
                <ReadoutRow label="P" value={`${readouts.P} Pa`} />
                <ReadoutRow label="U" value={`${readouts.U} J`} />
                <ReadoutRow label="S" value={`${readouts.S} J/K`} />
                <ReadoutRow label="N" value={readouts.N} />
                <ReadoutRow label="ρ_C" value={readouts.rhoC} />
                {params.mode === 'fluid' && (
                  <ReadoutRow label="μ" value={readouts.mu} />
                )}
              </div>
            </div>

            {/* Phase indicator */}
            <div>
              <h3 className="text-gray-400 text-xs font-bold uppercase tracking-wider mb-2">Phase</h3>
              <div className={`text-center py-2 rounded-lg text-xs font-bold ${
                params.networkDensity < 0.3
                  ? 'bg-blue-900/30 text-blue-300 border border-blue-800'
                  : params.networkDensity > 0.7
                  ? 'bg-orange-900/30 text-orange-300 border border-orange-800'
                  : 'bg-yellow-900/30 text-yellow-300 border border-yellow-800'
              }`}>
                {params.networkDensity < 0.3 ? 'GAS' :
                 params.networkDensity > 0.7 ? 'LIQUID' : 'TRANSITION'}
              </div>
            </div>
          </div>
        </div>

        {/* ── Status Bar ── */}
        <div className="h-8 bg-dark/90 border-t border-gray-800 flex items-center px-4 gap-4">
          <button
            onClick={() => setPlaying(!playing)}
            className="text-primaryDark text-xs font-mono hover:text-white transition-colors"
          >
            {playing ? '⏸ Pause' : '▶ Play'}
          </button>
          <span className="text-gray-600 text-xs">
            Spectral-native dynamics &middot; No backend &middot; Derived light
          </span>
        </div>
      </div>
    </>
  );
}
