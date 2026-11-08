import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { motion } from "framer-motion";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import useActiveSection from "@/components/landing/useActiveSection";

const DocsSimulation = dynamic(() => import("@/engine/DocsSimulation"), { ssr: false });

// Each section drives the simulation to demonstrate its concept.
const FLUID_SECTIONS = [
  {
    id: "transition",
    label: "ρ_C: Gas → Liquid",
    params: {
      temperature: 350, networkDensity: 0.5, volume: 0.85,
      curlFreq: 0.32, octaves: 2, turbulence: 0.3,
      colorA: "#7fc8e8", colorB: "#58e6d9",
    },
  },
  {
    id: "viscosity",
    label: "μ = τ_c × g",
    params: {
      temperature: 293, networkDensity: 0.85, volume: 0.7,
      curlFreq: 0.45, octaves: 4, turbulence: 0.4,
      colorA: "#58e6d9", colorB: "#88e6d9",
    },
  },
  {
    id: "navier",
    label: "Navier-Stokes",
    params: {
      temperature: 293, networkDensity: 0.9, volume: 0.7,
      curlFreq: 0.5, octaves: 4, turbulence: 0.5,
      colorA: "#58e6d9", colorB: "#a0e6d9",
      flowDir: [0.05, 0, 0],
    },
  },
  {
    id: "diffusion",
    label: "D = kBT / 6πμr",
    params: {
      temperature: 320, networkDensity: 0.8, volume: 0.85,
      curlFreq: 0.38, octaves: 3, turbulence: 0.6,
      colorA: "#a0e6d9", colorB: "#ffd9a0",
    },
  },
  {
    id: "phase",
    label: "Phase Transition",
    params: {
      temperature: 200, networkDensity: 0.95, volume: 0.55,
      curlFreq: 0.55, octaves: 4, turbulence: 0.3,
      colorA: "#58e6d9", colorB: "#7fc8e8",
    },
  },
  {
    id: "triple",
    label: "Triple Observation",
    params: {
      temperature: 350, networkDensity: 0.85, volume: 0.75,
      curlFreq: 0.42, octaves: 4, turbulence: 0.5,
      colorA: "#fff09e", colorB: "#58e6d9",
    },
  },
  {
    id: "poiseuille",
    label: "Poiseuille Flow",
    params: {
      temperature: 293, networkDensity: 0.9, volume: 0.8,
      curlFreq: 0.3, octaves: 3, turbulence: 0.2,
      colorA: "#58e6d9", colorB: "#88e6d9",
      flowDir: [0.15, 0, 0],
    },
  },
  {
    id: "turbulence",
    label: "Turbulence: Re > Re_c",
    params: {
      temperature: 600, networkDensity: 0.85, volume: 0.85,
      curlFreq: 0.6, octaves: 4, turbulence: 1.0,
      colorA: "#58e6d9", colorB: "#ffa464",
    },
  },
  {
    id: "light",
    label: "c = Δx / τ_c",
    params: {
      temperature: 400, networkDensity: 0.85, volume: 0.8,
      curlFreq: 0.4, octaves: 4, turbulence: 0.3,
      colorA: "#fff09e", colorB: "#58e6d9",
    },
  },
  {
    id: "usage",
    label: "Try It Yourself",
    params: {
      temperature: 293, networkDensity: 0.85, volume: 0.8,
      curlFreq: 0.4, octaves: 4, turbulence: 0.4,
      colorA: "#58e6d9", colorB: "#88e6d9",
    },
  },
];

function Equation({ children, label }) {
  return (
    <div className="my-6 bg-gray-900/60 border border-gray-800 rounded-lg p-4 text-center overflow-x-auto">
      <code className="text-primaryDark text-lg font-mono">{children}</code>
      {label && <div className="text-gray-500 text-xs mt-2 font-mono">{label}</div>}
    </div>
  );
}

function Section({ title, children, sectionRef }) {
  return (
    <motion.section
      ref={sectionRef}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      viewport={{ once: true }}
      className="mb-24 min-h-[60vh]"
    >
      <h2 className="text-2xl font-bold text-light mb-6 border-b border-gray-800 pb-3">{title}</h2>
      <div className="text-gray-300 leading-relaxed space-y-4">{children}</div>
    </motion.section>
  );
}

function ValidationTable({ data }) {
  return (
    <div className="overflow-x-auto my-6">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700">
            {Object.keys(data[0]).map((key) => (
              <th key={key} className="text-left text-gray-400 py-2 px-3 font-mono text-xs uppercase">{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-900/30">
              {Object.values(row).map((val, j) => (
                <td key={j} className="py-2 px-3 text-gray-300 font-mono text-xs">{val}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function FluidsPage() {
  const sectionRefs = FLUID_SECTIONS.map(() => useRef(null));
  const activeIdx = useActiveSection(sectionRefs, 0);
  const activeSection = FLUID_SECTIONS[activeIdx];

  return (
    <>
      <Head>
        <title>Fluid Dynamics — Maupertuis</title>
        <meta name="description" content="Spectral-native fluid dynamics: viscosity, Navier-Stokes, diffusion, phase transitions from partition network interference." />
      </Head>
      <TransitionEffect />

      <main className="bg-dark text-light min-h-screen">
        <Layout className="!pt-8">
          {/* Header */}
          <div className="mb-12 max-w-3xl">
            <p className="text-primaryDark text-sm font-mono tracking-widest uppercase mb-3">Fluid Dynamics</p>
            <h1 className="text-4xl font-bold mb-6 lg:text-3xl">
              Viscous Flow from Partition Network Interference
            </h1>
            <p className="text-gray-400 text-lg">
              The Navier-Stokes equations, viscosity, diffusion, heat conduction, and phase transitions —
              all emerging from dense partition networks. Viscosity is not phenomenological: it is
              μ = τ_c × g.
            </p>
            <div className="mt-6 flex gap-4">
              <Link href="/simulate" className="px-6 py-2.5 bg-primaryDark text-dark font-bold rounded-lg hover:bg-primaryDark/80 transition-colors text-sm">
                Open Fluid Simulator
              </Link>
            </div>
          </div>

          {/* ── 2-column layout ── */}
          <div className="grid grid-cols-2 gap-12 lg:grid-cols-1">
            {/* Left: text */}
            <div className="min-w-0">

              <Section title="From Gas to Liquid: Network Density" sectionRef={sectionRefs[0]}>
                <p>
                  The same spectral oscillators that behave as gas particles at low density behave as
                  fluid elements at high density. The transition is controlled by a single parameter:
                  network density ρ_C.
                </p>
                <Equation label="Network density">
                  {"ρ_C = |{(p,q) : d_S(p,q) < r_c}| / (N(N-1)/2)"}
                </Equation>
                <ul className="list-disc list-inside space-y-2 text-gray-400 ml-4">
                  <li><span className="text-blue-300">ρ_C &lt; 0.3</span> — Gas: sparse connectivity, rare collisions</li>
                  <li><span className="text-yellow-300">0.3 ≤ ρ_C ≤ 0.7</span> — Transition: percolation threshold</li>
                  <li><span className="text-orange-300">ρ_C &gt; 0.7</span> — Liquid: dense connectivity, persistent coupling</li>
                </ul>
                <p>
                  Watch the visualization on the right shift from open swirls (gas regime) to tight
                  braided eddies as we cross the percolation threshold.
                </p>
              </Section>

              <Section title="Viscosity = Partition Lag × Coupling Strength" sectionRef={sectionRefs[1]}>
                <p>
                  Viscosity is not a phenomenological parameter — it is derived from two microscopic
                  quantities:
                </p>
                <Equation label="Viscosity from partition parameters">
                  {"μ = τ_c × g"}
                </Equation>
                <p>
                  <strong className="text-primaryDark">τ_c</strong> is the partition lag — the time for each
                  molecule to complete a categorical state transition. For water at 20°C: τ_c ≈ 0.15 ps.
                </p>
                <p>
                  <strong className="text-primaryDark">g</strong> is the coupling strength — the force gradient
                  of the intermolecular potential at equilibrium separation. For water: g ≈ 6.6 N/m.
                </p>
                <ValidationTable data={[
                  { Fluid: "Water", "τ_c (ps)": "0.15", "g (N/m)": "6.6", "μ_pred": "0.99", "μ_exp": "1.00", Error: "1.2%" },
                  { Fluid: "Ethanol", "τ_c (ps)": "0.22", "g (N/m)": "5.1", "μ_pred": "1.12", "μ_exp": "1.07", Error: "4.7%" },
                  { Fluid: "Acetone", "τ_c (ps)": "0.12", "g (N/m)": "2.6", "μ_pred": "0.31", "μ_exp": "0.32", Error: "3.1%" },
                  { Fluid: "Glycerol", "τ_c (ps)": "2.80", "g (N/m)": "334", "μ_pred": "935", "μ_exp": "934", Error: "0.1%" },
                ]} />
                <p className="text-gray-500 text-sm">Mean absolute error: 2.9% across 12 pure liquids.</p>
              </Section>

              <Section title="Navier-Stokes from Kirchhoff" sectionRef={sectionRefs[2]}>
                <p>
                  The partition network is a circuit. Each node carries a chemical potential (voltage).
                  Each edge carries a flux (current). Kirchhoff&apos;s laws apply exactly:
                </p>
                <ul className="list-disc list-inside space-y-3 text-gray-400 ml-4">
                  <li>
                    <span className="text-gray-200">KCL → Continuity:</span>{" "}
                    Conservation of categorical states yields ∂ρ/∂t + ∇·(ρv) = 0
                  </li>
                  <li>
                    <span className="text-gray-200">KVL → Pressure field:</span>{" "}
                    Single-valued potential ensures pressure is a well-defined scalar
                  </li>
                  <li>
                    <span className="text-gray-200">Viscous term:</span>{" "}
                    Collective lag of partition operations propagates as μ∇²v
                  </li>
                </ul>
                <Equation label="Navier-Stokes (continuum limit of Kirchhoff)">
                  {"ρ Dv/Dt = −∇p + μ∇²v + f"}
                </Equation>
                <p>
                  The visualization shows directional flow with viscous coupling — adjacent particles
                  drag each other along, exactly as the μ∇²v term requires.
                </p>
              </Section>

              <Section title="Diffusion and Heat Conduction" sectionRef={sectionRefs[3]}>
                <p>
                  Fick&apos;s law emerges from partition dynamics: a concentration gradient creates
                  imbalance in categorical state occupancy.
                </p>
                <Equation label="Stokes-Einstein diffusion">
                  {"D = kBT / (6π μ r) = kBT / (6π (τ_c × g) r)"}
                </Equation>
                <p>
                  Fourier&apos;s law emerges similarly: temperature differences create partition lag
                  gradients, and energy flows from fast-τ_c to slow-τ_c regions:
                </p>
                <Equation label="Thermal conductivity">
                  {"κ = kB / τ_c × N/V"}
                </Equation>
              </Section>

              <Section title="Phase Transitions" sectionRef={sectionRefs[4]}>
                <p>
                  The gas-to-liquid transition is not imposed — it emerges from network topology.
                  As compression increases ρ_C past the percolation threshold (≈ 0.5), viscosity
                  jumps by orders of magnitude:
                </p>
                <ValidationTable data={[
                  { "V/V₀": "1.0", "ρ_C": "0.05", "μ (mPa·s)": "~0.02", Phase: "Gas" },
                  { "V/V₀": "0.5", "ρ_C": "0.15", "μ (mPa·s)": "~0.05", Phase: "Gas" },
                  { "V/V₀": "0.2", "ρ_C": "0.42", "μ (mPa·s)": "~0.3", Phase: "Transition" },
                  { "V/V₀": "0.1", "ρ_C": "0.78", "μ (mPa·s)": "~1.0", Phase: "Liquid" },
                  { "V/V₀": "0.05", "ρ_C": "0.95", "μ (mPa·s)": "~5.0", Phase: "Dense liquid" },
                ]} />
                <p>
                  The visualization compresses tightly — particles forced into a small volume couple
                  strongly, viscosity rises, motion becomes coherent.
                </p>
              </Section>

              <Section title="Triple Observation" sectionRef={sectionRefs[5]}>
                <p>
                  The ray march through the fluid volume computes three observations simultaneously
                  at each step — unified by the partition state σ(r):
                </p>
                <Equation label="Triple Observation Identity">
                  {"μ_abs(r) = κ₁ / (τ_c · d_S) = κ₂ · G(r) · RT"}
                </Equation>
                <ul className="list-disc list-inside space-y-2 text-gray-400 ml-4">
                  <li><span className="text-gray-200">Optical</span> — Beer-Lambert + Mie scattering + refraction</li>
                  <li><span className="text-gray-200">Chromatographic</span> — fluid IS a 3D column; retention from S-distance</li>
                  <li><span className="text-gray-200">Circuit</span> — Kirchhoff current IS the Stokes velocity field</li>
                </ul>
                <p className="mt-4">
                  Velocity is recovered from Doppler-shifted retention anisotropy: opposing rays
                  measure different effective S-distances, encoding the local velocity projection.
                </p>
              </Section>

              <Section title="Poiseuille Flow Validation" sectionRef={sectionRefs[6]}>
                <p>
                  For pressure-driven flow through a cylindrical channel, the parabolic velocity
                  profile is recovered from ray-marched retention anisotropy:
                </p>
                <ValidationTable data={[
                  { "r/R": "0.0", "Predicted": "1.000", "Recovered": "0.998", Error: "0.2%" },
                  { "r/R": "0.25", "Predicted": "0.938", "Recovered": "0.935", Error: "0.3%" },
                  { "r/R": "0.50", "Predicted": "0.750", "Recovered": "0.748", Error: "0.3%" },
                  { "r/R": "0.75", "Predicted": "0.438", "Recovered": "0.441", Error: "0.7%" },
                  { "r/R": "0.95", "Predicted": "0.098", "Recovered": "0.101", Error: "3.1%" },
                ]} />
                <p className="text-gray-500 text-sm">Mean error: 0.9%</p>
                <p>
                  Watch the visualization: particles drift in a coherent direction — the directional
                  flow component characteristic of pressure-driven flow.
                </p>
              </Section>

              <Section title="Turbulence" sectionRef={sectionRefs[7]}>
                <p>
                  Both laminar and turbulent flow are governed by the same partition network dynamics.
                  The difference is the partition lag spectrum:
                </p>
                <ul className="list-disc list-inside space-y-2 text-gray-400 ml-4">
                  <li><span className="text-blue-300">Laminar</span> — narrow P(τ_c), single timescale</li>
                  <li><span className="text-orange-300">Turbulent</span> — broad P(τ_c), multi-scale vortices</li>
                </ul>
                <p>
                  The critical Reynolds number Re_c corresponds to the onset of spectral broadening.
                  The visualization here is at high turbulence — chaotic, multi-scale eddies forming
                  and dissipating.
                </p>
              </Section>

              <Section title="Derived Light for Fluid Observation" sectionRef={sectionRefs[8]}>
                <p>
                  As in the gas case, light is derived from partition operations:
                </p>
                <Equation label="Speed of light">{"c = Δx / τ_c = 2.995 × 10⁸ m/s"}</Equation>
                <p>
                  In the fluid ray march, this derived light refracts through density gradients
                  (Snell&apos;s discrete law from RPI). The refractive index is proportional to
                  network density:
                </p>
                <Equation label="Refractive index">{"n(r) = 1 + α × ρ_C(r)"}</Equation>
                <p>
                  Scattering transitions from Rayleigh (gas) to Mie (liquid) as density increases.
                  More scattering = higher-rank transfer matrix = better observation.
                </p>
              </Section>

              <Section title="Using the Fluid Simulator" sectionRef={sectionRefs[9]}>
                <p>In the simulator, set Network Density above 0.7 for liquid behaviour. The key controls:</p>
                <ul className="list-disc list-inside space-y-2 text-gray-400 ml-4">
                  <li><span className="text-gray-200">Network Density</span> (0.7–1.0) — controls coupling, viscosity, refraction</li>
                  <li><span className="text-gray-200">Temperature</span> (50–2000 K) — controls τ_c via Arrhenius</li>
                  <li><span className="text-gray-200">Volume</span> — controls compression; smaller = denser network</li>
                  <li><span className="text-gray-200">Particles</span> — more particles = richer network topology</li>
                </ul>
                <p className="mt-4">
                  Use the presets: &ldquo;Water (20°C)&rdquo;, &ldquo;Ethanol (20°C)&rdquo;,
                  &ldquo;Glycerol (20°C)&rdquo; for pre-configured fluid simulations.
                </p>
                <div className="mt-6">
                  <Link href="/simulate" className="inline-block px-6 py-3 bg-primaryDark text-dark font-bold rounded-lg hover:bg-primaryDark/80 transition-colors text-sm">
                    Launch Full Simulator →
                  </Link>
                </div>
              </Section>

            </div>

            {/* Right: simulation (sticky) */}
            <div className="lg:hidden">
              <div className="sticky top-8 h-[calc(100vh-4rem)]">
                <div className="relative w-full h-full bg-gradient-to-br from-gray-900 to-dark rounded-2xl overflow-hidden border border-gray-800">
                  <Canvas
                    camera={{ position: [0, 0, 4.5], fov: 50 }}
                    gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
                    dpr={[1, 1.5]}
                    style={{ background: 'transparent' }}
                  >
                    <color attach="background" args={['#0a0a0f']} />
                    <DocsSimulation params={activeSection.params} />
                  </Canvas>

                  {/* Section indicator */}
                  <div className="absolute bottom-4 left-4 right-4 flex flex-col gap-2 pointer-events-none">
                    <div className="bg-dark/80 backdrop-blur-sm rounded-lg px-3 py-2 border border-gray-800">
                      <div className="text-primaryDark text-[10px] font-mono uppercase tracking-wider mb-1">
                        Section {activeIdx + 1} / {FLUID_SECTIONS.length}
                      </div>
                      <div className="text-gray-200 text-sm font-mono">
                        {activeSection.label}
                      </div>
                    </div>

                    {/* Live params */}
                    <div className="grid grid-cols-3 gap-2 text-[10px] font-mono">
                      <div className="bg-dark/60 backdrop-blur-sm rounded px-2 py-1 border border-gray-800">
                        <div className="text-gray-500">T</div>
                        <div className="text-primaryDark">{activeSection.params.temperature} K</div>
                      </div>
                      <div className="bg-dark/60 backdrop-blur-sm rounded px-2 py-1 border border-gray-800">
                        <div className="text-gray-500">V</div>
                        <div className="text-primaryDark">{activeSection.params.volume.toFixed(2)}</div>
                      </div>
                      <div className="bg-dark/60 backdrop-blur-sm rounded px-2 py-1 border border-gray-800">
                        <div className="text-gray-500">ρ_C</div>
                        <div className="text-primaryDark">{activeSection.params.networkDensity.toFixed(2)}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Layout>
      </main>
    </>
  );
}
