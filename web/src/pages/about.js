import Head from "next/head";
import Link from "next/link";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";

function PaperCard({ title, subtitle, description, href }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      viewport={{ once: true }}
      className="bg-gray-900/40 border border-gray-800 rounded-xl p-6 hover:border-primaryDark/30 transition-colors"
    >
      <p className="text-primaryDark text-xs font-mono uppercase tracking-wider mb-2">{subtitle}</p>
      <h3 className="text-light text-lg font-bold mb-3">{title}</h3>
      <p className="text-gray-400 text-sm leading-relaxed mb-4">{description}</p>
      {href && (
        <Link href={href} className="text-primaryDark text-sm font-mono hover:underline">
          Read more →
        </Link>
      )}
    </motion.div>
  );
}

export default function AboutPage() {
  return (
    <>
      <Head>
        <title>About — Maupertuis</title>
        <meta name="description" content="The research behind Maupertuis: spectral-native gas and fluid dynamics from bounded phase space geometry." />
      </Head>
      <TransitionEffect />

      <main className="bg-dark text-light min-h-screen">
        <Layout className="!pt-8">
          <div className="max-w-4xl mx-auto">

            {/* Header */}
            <div className="mb-16">
              <p className="text-primaryDark text-sm font-mono tracking-widest uppercase mb-3">About</p>
              <h1 className="text-4xl font-bold mb-6">The Research Behind Maupertuis</h1>
              <p className="text-gray-400 text-lg max-w-2xl">
                Maupertuis is named after the Principle of Least Action. The tool implements
                a theoretical framework where all dynamics is partition depth minimization —
                the categorical analogue of least action in bounded phase space.
              </p>
            </div>

            {/* Core Idea */}
            <motion.section
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
              className="mb-16 bg-gray-900/30 border border-gray-800 rounded-2xl p-8"
            >
              <h2 className="text-2xl font-bold mb-4">The Core Idea</h2>
              <p className="text-gray-300 leading-relaxed mb-4">
                Every processor is an oscillator. Every oscillator has a spectrum. Every spectrum
                encodes a molecular identity. Spectral interference computes dynamics. Derived light
                enables observation. The ray march IS measurement.
              </p>
              <p className="text-gray-300 leading-relaxed mb-4">
                This chain of identities eliminates the representation bottleneck that has defined
                computational fluid dynamics since its inception. There is no mesh, no numerical
                integration of equations of motion, no pre-stored molecular data, and no assumed light.
                Everything is derived from a single axiom: the Bounded Phase Space Law.
              </p>
              <p className="text-gray-400 text-sm">
                The framework is validated with zero adjustable parameters against experimental data
                spanning ideal gas laws, Maxwell-Boltzmann distributions, viscosity predictions for
                12 pure liquids, Poiseuille flow, and gas-to-liquid phase transitions.
              </p>
            </motion.section>

            {/* Papers */}
            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-8">Foundational Papers</h2>
              <div className="grid grid-cols-2 gap-6 lg:grid-cols-1">
                <PaperCard
                  subtitle="Gas Dynamics"
                  title="Spectral-Native Gas Dynamics"
                  description="Real-time kinetic gas behaviour from hardware oscillation interference with ray-marched observation. Ideal gas law, Maxwell-Boltzmann, equipartition, adiabatic processes."
                  href="/gas"
                />
                <PaperCard
                  subtitle="Fluid Dynamics"
                  title="Spectral-Native Fluid Dynamics"
                  description="Real-time viscous flow from partition network interference with ray-marched observation. Viscosity μ = τ_c × g, Navier-Stokes from Kirchhoff, phase transitions."
                  href="/fluids"
                />
                <PaperCard
                  subtitle="Foundation"
                  title="On the Thermodynamic Consequences of Bounded Phase Space"
                  description="Proves that gas thermodynamics is computationally equivalent to trajectory completion in bounded S-entropy space. Temperature IS processing rate. Entropy IS complexity."
                />
                <PaperCard
                  subtitle="Foundation"
                  title="The Gas Particle from First Principles"
                  description="Derives the complete structure of a gas particle from the Bounded Phase Space Law alone: partition coordinates, shell capacity C(n) = 2n², five dynamical theorems."
                />
                <PaperCard
                  subtitle="Spectral Matching"
                  title="Universal Spectral Matching"
                  description="Proves all comparison reduces to computer vision through oscillatory representation and GPU-parallel interference. Five-pass shader pipeline."
                />
                <PaperCard
                  subtitle="Mass Transfer"
                  title="Partition Operations in Fluid Flux Mechanisms"
                  description="Derives viscosity, light, and chromatographic retention from a single partition lag parameter τ_c. Validates across 12 liquids, 15 pharmaceuticals, and 20 UV-Vis spectra."
                />
                <PaperCard
                  subtitle="Ray Tracing"
                  title="Multi-Modal Ray-Tracing as Cellular Computation"
                  description="A single ray march simultaneously computes optical, chromatographic, and circuit observations. Eight oscillator classes. Triple Observation Identity."
                />
                <PaperCard
                  subtitle="Harmonic Networks"
                  title="Harmonic Molecular Resonator"
                  description="Light circulates in closed loops within molecules without walls. Virtual resonant cavities from categorical coupling. Self-clocking, self-validating networks."
                />
              </div>
            </section>

            {/* How to Use */}
            <section className="mb-16">
              <h2 className="text-2xl font-bold mb-6">How to Use the Simulator</h2>
              <div className="space-y-6">
                <div className="flex gap-4 items-start">
                  <div className="w-8 h-8 rounded-full bg-primaryDark/20 border border-primaryDark/40 flex items-center justify-center text-primaryDark font-bold text-sm shrink-0">1</div>
                  <div>
                    <h3 className="text-light font-bold mb-1">Choose a mode</h3>
                    <p className="text-gray-400 text-sm">Gas (sparse network, kinetic theory) or Fluid (dense network, Navier-Stokes). The mode toggle is in the toolbar.</p>
                  </div>
                </div>
                <div className="flex gap-4 items-start">
                  <div className="w-8 h-8 rounded-full bg-primaryDark/20 border border-primaryDark/40 flex items-center justify-center text-primaryDark font-bold text-sm shrink-0">2</div>
                  <div>
                    <h3 className="text-light font-bold mb-1">Select a preset or adjust parameters</h3>
                    <p className="text-gray-400 text-sm">Presets configure realistic conditions (Water at 20°C, Ideal Gas at 300K, etc.). Or use the sliders: temperature, volume, particles, network density.</p>
                  </div>
                </div>
                <div className="flex gap-4 items-start">
                  <div className="w-8 h-8 rounded-full bg-primaryDark/20 border border-primaryDark/40 flex items-center justify-center text-primaryDark font-bold text-sm shrink-0">3</div>
                  <div>
                    <h3 className="text-light font-bold mb-1">Observe</h3>
                    <p className="text-gray-400 text-sm">The volumetric ray march renders the gas/fluid in real time. Drag to orbit the camera. The side panel shows live thermodynamic readouts (T, P, U, μ).</p>
                  </div>
                </div>
                <div className="flex gap-4 items-start">
                  <div className="w-8 h-8 rounded-full bg-primaryDark/20 border border-primaryDark/40 flex items-center justify-center text-primaryDark font-bold text-sm shrink-0">4</div>
                  <div>
                    <h3 className="text-light font-bold mb-1">Experiment</h3>
                    <p className="text-gray-400 text-sm">Raise temperature → watch emission brighten. Compress volume → watch density increase. Push network density past 0.7 → watch the phase transition from gas to liquid.</p>
                  </div>
                </div>
              </div>
            </section>

            {/* Author */}
            <section className="mb-16 text-center">
              <p className="text-gray-500 text-sm">
                Developed by Kundai Farai Sachikonye · AIMe Registry for Artificial Intelligence
              </p>
            </section>

          </div>
        </Layout>
      </main>
    </>
  );
}
