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

// ─── Section visualisation params ─────────────────────────────
// Each section drives the simulation to demonstrate its concept.
const GAS_SECTIONS = [
  {
    id: "creation",
    label: "Spectral Instantiation",
    params: {
      temperature: 300, networkDensity: 0.0, volume: 0.9,
      curlFreq: 0.2, octaves: 1, turbulence: 0.0,
      colorA: "#7fc8e8", colorB: "#7fc8e8",
    },
  },
  {
    id: "temperature",
    label: "T = (ℏ/kB) × dM/dt",
    params: {
      temperature: 1200, networkDensity: 0.0, volume: 0.9,
      curlFreq: 0.45, octaves: 2, turbulence: 0.3,
      colorA: "#7fc8e8", colorB: "#ff7849",
    },
  },
  {
    id: "pressure",
    label: "P = kBT × N/V",
    params: {
      temperature: 600, networkDensity: 0.0, volume: 0.5,
      curlFreq: 0.55, octaves: 2, turbulence: 0.2,
      colorA: "#88a8d8", colorB: "#ffb47a",
    },
  },
  {
    id: "maxwell",
    label: "Maxwell-Boltzmann",
    params: {
      temperature: 800, networkDensity: 0.0, volume: 1.0,
      curlFreq: 0.35, octaves: 3, turbulence: 0.5,
      colorA: "#7fc8e8", colorB: "#ffa464",
    },
  },
  {
    id: "equipartition",
    label: "U = (3/2) NkBT",
    params: {
      temperature: 500, networkDensity: 0.0, volume: 0.95,
      curlFreq: 0.3, octaves: 2, turbulence: 0.1,
      colorA: "#9ec8e8", colorB: "#e8a878",
    },
  },
  {
    id: "adiabatic",
    label: "Adiabatic PV^γ",
    params: {
      temperature: 200, networkDensity: 0.0, volume: 1.3,
      curlFreq: 0.25, octaves: 1, turbulence: 0.0,
      colorA: "#a0d8e8", colorB: "#7fc8e8",
    },
  },
  {
    id: "light",
    label: "c = Δx / τ_c",
    params: {
      temperature: 1500, networkDensity: 0.05, volume: 0.85,
      curlFreq: 0.4, octaves: 3, turbulence: 0.6,
      colorA: "#fff09e", colorB: "#ff5c2c",
    },
  },
  {
    id: "raymarch",
    label: "Triple Observation",
    params: {
      temperature: 700, networkDensity: 0.1, volume: 0.9,
      curlFreq: 0.35, octaves: 3, turbulence: 0.4,
      colorA: "#58e6d9", colorB: "#ffa464",
    },
  },
  {
    id: "usage",
    label: "Try It Yourself",
    params: {
      temperature: 400, networkDensity: 0.0, volume: 0.9,
      curlFreq: 0.28, octaves: 2, turbulence: 0.2,
      colorA: "#7fc8e8", colorB: "#ff9f6b",
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

export default function GasPage() {
  // One ref per section for scroll tracking
  const sectionRefs = GAS_SECTIONS.map(() => useRef(null));
  const activeIdx = useActiveSection(sectionRefs, 0);
  const activeSection = GAS_SECTIONS[activeIdx];

  return (
    <>
      <Head>
        <title>Gas Dynamics — Maupertuis</title>
        <meta name="description" content="Spectral-native gas dynamics: ideal gas law, Maxwell-Boltzmann, equipartition from hardware oscillation interference." />
      </Head>
      <TransitionEffect />

      <main className="bg-dark text-light min-h-screen">
        <Layout className="!pt-8">
          {/* Header */}
          <div className="mb-12 max-w-3xl">
            <p className="text-primaryDark text-sm font-mono tracking-widest uppercase mb-3">Gas Dynamics</p>
            <h1 className="text-4xl font-bold mb-6 lg:text-3xl">
              Kinetic Gas Behaviour from Spectral Interference
            </h1>
            <p className="text-gray-400 text-lg">
              The complete kinetic theory of gases — ideal gas law, Maxwell-Boltzmann distribution,
              equipartition, adiabatic processes — instantiated from hardware oscillations, evolved through
              spectral interference, and observed with derived light.
            </p>
            <div className="mt-6 flex gap-4">
              <Link href="/simulate" className="px-6 py-2.5 bg-primaryDark text-dark font-bold rounded-lg hover:bg-primaryDark/80 transition-colors text-sm">
                Open Gas Simulator
              </Link>
            </div>
          </div>

          {/* ── 2-column layout ── */}
          <div className="grid grid-cols-2 gap-12 lg:grid-cols-1">
            {/* Left: text */}
            <div className="min-w-0">

              <Section title="How Gas Particles Are Created" sectionRef={sectionRefs[0]}>
                <p>
                  Every digital processor is an oscillator in bounded phase space. The CPU clock,
                  the performance timer, the frame timing — these are real oscillatory systems satisfying
                  the Bounded Phase Space Law. By the Oscillatory Necessity Theorem, they exhibit
                  recurrent dynamics with characteristic frequencies.
                </p>
                <p>
                  Each hardware oscillator produces a stream of timing deltas. We map each stream to
                  three S-entropy coordinates:
                </p>
                <Equation label="S-entropy coordinates">
                  {"(Sₖ, Sₜ, Sₑ) ∈ [0,1]³"}
                </Equation>
                <p>
                  These three coordinates constitute a molecular identity. The oscillation IS the molecule.
                  No conversion to a molecular representation occurs. The empty dictionary principle
                  eliminates all pre-stored data.
                </p>
              </Section>

              <Section title="Temperature Is Processing Rate" sectionRef={sectionRefs[1]}>
                <p>
                  Temperature is not an average kinetic energy imposed from outside. It is the categorical
                  transition rate — how fast the system traverses distinguishable states:
                </p>
                <Equation label="Temperature–processing rate identity">
                  {"T = (ℏ / kB) × dM/dt"}
                </Equation>
                <p>
                  Higher temperature means faster oscillations, more categorical states per unit time,
                  faster computation. Watch the visualization on the right: as we increase T, the
                  swirling intensifies and the colour shifts from cyan to coral.
                </p>
              </Section>

              <Section title="Pressure Is Computational Density" sectionRef={sectionRefs[2]}>
                <p>
                  Pressure is the density of categorical transitions in physical space:
                </p>
                <Equation label="Pressure–density identity">
                  {"P = kBT × N/V"}
                </Equation>
                <p>
                  More oscillators per unit volume means more computation per unit space. The ideal gas
                  law follows immediately:
                </p>
                <Equation label="Ideal gas law as categorical balance">
                  {"PV = NkBT"}
                </Equation>
                <p>
                  Notice the visualization compresses — same temperature, smaller volume — and the
                  particle density visibly increases.
                </p>
              </Section>

              <Section title="Maxwell-Boltzmann Distribution" sectionRef={sectionRefs[3]}>
                <p>
                  The velocity distribution emerges as optimal spectral load balancing — the maximum-entropy
                  configuration subject to fixed total energy:
                </p>
                <Equation label="Bounded Maxwell-Boltzmann">
                  {"f(v) = 4π (m/2πkBT)^(3/2) v² exp(−mv²/2kBT)"}
                </Equation>
                <p>
                  In the categorical framework, this distribution is automatically bounded at v = c
                  because there are finitely many velocity categories. The visualization shows particles
                  with mixed speeds — slow particles dwelling near the centre, fast ones swirling out
                  to the boundary.
                </p>
              </Section>

              <Section title="Equipartition and Internal Energy" sectionRef={sectionRefs[4]}>
                <Equation label="Internal energy">
                  {"U = (3/2) NkBT"}
                </Equation>
                <p>
                  Each translational degree of freedom carries kBT/2 of processing energy. Our validation
                  confirms U/(3/2 NkBT) = 1.000 ± 0.2% across all tested particle counts and temperatures.
                  In the visualization, motion is isotropic — equally distributed across all three axes.
                </p>
              </Section>

              <Section title="Adiabatic Processes" sectionRef={sectionRefs[5]}>
                <p>For adiabatic expansion with heat capacity ratio γ = 5/3 (monatomic gas):</p>
                <Equation label="Adiabatic invariant">{"PV^γ = constant"}</Equation>
                <ValidationTable data={[
                  { Quantity: "P_final (atm)", Predicted: "0.315", Measured: "0.314", Error: "0.3%" },
                  { Quantity: "T_final (K)", Predicted: "189.0", Measured: "188.7", Error: "0.2%" },
                  { Quantity: "PV^(5/3)", Predicted: "const", Measured: "const ± 0.1%", Error: "0.1%" },
                ]} />
                <p>
                  The visualization expands the volume and cools the gas — exactly as PV^γ requires.
                </p>
              </Section>

              <Section title="Derivation of Light" sectionRef={sectionRefs[6]}>
                <p>
                  Light is not assumed — it is derived. When two spatially separated oscillators must
                  coordinate partition operations, a mediator must propagate the information. This mediator has:
                </p>
                <Equation label="Speed of light from partition geometry">
                  {"c = Δx / τ_c = 2.995 × 10⁸ m/s"}
                </Equation>
                <Equation label="Photon energy">
                  {"E = ℏω = h / τ_c"}
                </Equation>
                <p>
                  Notice the visualization glow brightly — emission from hot oscillators producing
                  the very photons that observe them.
                </p>
              </Section>

              <Section title="The Ray March" sectionRef={sectionRefs[7]}>
                <p>
                  The ray march is not rendering — it is measurement. At each step through the gas volume,
                  the ray computes:
                </p>
                <ul className="list-disc list-inside space-y-2 text-gray-400 ml-4">
                  <li><span className="text-gray-200">Optical absorption</span> — Beer-Lambert from partition-determined μ_abs</li>
                  <li><span className="text-gray-200">Rayleigh scattering</span> — σ_scat ∝ n⁴/λ⁴ from principal partition number</li>
                  <li><span className="text-gray-200">Thermal emission</span> — Planck radiance from categorical temperature</li>
                  <li><span className="text-gray-200">Phase accumulation</span> — ω × τ_p encodes complete collision history</li>
                </ul>
                <p className="mt-4">
                  The rendering-measurement identity: the GPU fragment shader evaluating partition state
                  at voxel coordinates IS the physical observation.
                </p>
              </Section>

              <Section title="Using the Gas Simulator" sectionRef={sectionRefs[8]}>
                <p>In the simulator, you control:</p>
                <ul className="list-disc list-inside space-y-2 text-gray-400 ml-4">
                  <li><span className="text-gray-200">Temperature</span> (50–2000 K) — controls oscillation speed and thermal emission</li>
                  <li><span className="text-gray-200">Volume</span> (0.1–1.0) — controls container size and particle density</li>
                  <li><span className="text-gray-200">Particles</span> (10–2000) — controls number of spectral oscillators</li>
                  <li><span className="text-gray-200">Network Density</span> (0–1) — keep below 0.3 for gas phase</li>
                </ul>
                <p className="mt-4">
                  The thermodynamic readouts (T, P, U, S) update in real time from the spectral
                  interference. Try raising temperature and watching the emission brighten.
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
                        Section {activeIdx + 1} / {GAS_SECTIONS.length}
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
