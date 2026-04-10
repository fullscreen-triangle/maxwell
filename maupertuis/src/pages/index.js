import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import { Suspense } from "react";
import { motion } from "framer-motion";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";

const FluidModel = dynamic(() => import("@/components/landing/FluidModel"), {
  ssr: false,
});

function FeatureCard({ title, description, icon }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      viewport={{ once: true }}
      className="bg-dark/50 border border-gray-800 rounded-xl p-6 hover:border-primaryDark/40 transition-colors"
    >
      <div className="text-3xl mb-3">{icon}</div>
      <h3 className="text-light text-lg font-bold mb-2">{title}</h3>
      <p className="text-gray-400 text-sm leading-relaxed">{description}</p>
    </motion.div>
  );
}

export default function Home() {
  return (
    <>
      <Head>
        <title>Maupertuis — Spectral-Native Gas &amp; Fluid Dynamics</title>
        <meta
          name="description"
          content="Real-time gas and fluid dynamics from spectral interference. No backend. No simulation. Real molecules from hardware oscillations."
        />
      </Head>

      <TransitionEffect />

      <div className="min-h-screen bg-dark text-light">
        {/* ── Hero ── */}
        <section className="relative min-h-screen flex items-center">
          <Layout className="!pt-0">
            <div className="flex w-full items-center justify-between gap-8 md:flex-col">
              {/* Left: 3D Model */}
              <div className="w-1/2 h-[500px] md:w-full md:h-[350px] relative">
                <Canvas
                  camera={{ position: [3, 2, 4], fov: 40 }}
                  gl={{ antialias: true, alpha: true }}
                  style={{ background: 'transparent' }}
                >
                  <ambientLight intensity={0.4} />
                  <directionalLight position={[5, 5, 5]} intensity={1.2} color="#58E6D9" />
                  <directionalLight position={[-3, 2, -2]} intensity={0.5} color="#B63E96" />
                  <pointLight position={[0, 3, 0]} intensity={0.8} color="#ffffff" />
                  <Suspense fallback={null}>
                    <FluidModel scale={1.5} position={[0, -0.5, 0]} />
                    <Environment preset="city" />
                  </Suspense>
                  <OrbitControls
                    enableZoom={false}
                    enablePan={false}
                    autoRotate
                    autoRotateSpeed={0.5}
                    minPolarAngle={Math.PI / 4}
                    maxPolarAngle={Math.PI / 1.8}
                  />
                </Canvas>
              </div>

              {/* Right: Copy */}
              <div className="w-1/2 flex flex-col md:w-full md:text-center md:items-center">
                <motion.div
                  initial={{ opacity: 0, x: 30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                >
                  <p className="text-primaryDark text-sm font-mono tracking-widest uppercase mb-4">
                    Spectral-Native Dynamics
                  </p>
                  <h1 className="text-5xl font-bold leading-tight mb-6 xl:text-4xl lg:text-3xl">
                    Real-Time Fluid Dynamics.{" "}
                    <span className="text-primaryDark">No Server.</span>{" "}
                    <span className="text-gray-400">No Simulation.</span>
                  </h1>
                  <p className="text-gray-400 text-lg mb-8 max-w-lg md:text-base">
                    Gas and liquid behaviour from hardware oscillation interference.
                    Every molecule is real. Every interaction is spectral.
                    Runs entirely in your browser.
                  </p>
                  <div className="flex gap-4 md:justify-center">
                    <Link
                      href="/simulate"
                      className="px-8 py-3 bg-primaryDark text-dark font-bold rounded-lg
                                 hover:bg-primaryDark/80 transition-colors text-base"
                    >
                      Launch Simulator
                    </Link>
                    <Link
                      href="#features"
                      className="px-8 py-3 border border-gray-600 text-gray-300 rounded-lg
                                 hover:border-primaryDark hover:text-primaryDark transition-colors text-base"
                    >
                      Learn More
                    </Link>
                  </div>
                </motion.div>
              </div>
            </div>
          </Layout>
        </section>

        {/* ── Features ── */}
        <section id="features" className="py-24 border-t border-gray-800/50">
          <Layout>
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="text-center mb-16"
            >
              <p className="text-primaryDark text-sm font-mono tracking-widest uppercase mb-3">
                How It Works
              </p>
              <h2 className="text-3xl font-bold">
                Not a Simulation. An Instantiation.
              </h2>
            </motion.div>

            <div className="grid grid-cols-4 gap-6 lg:grid-cols-2 md:grid-cols-1">
              <FeatureCard
                icon="~"
                title="Hardware Oscillations"
                description="CPU clock cycles, timer counters, and frame timing ARE oscillatory systems. We map them directly to molecular spectral coordinates."
              />
              <FeatureCard
                icon="{"
                title="Spectral Interference"
                description="Molecules interact through spectral interference — no force calculations, no numerical integration. The interference IS the dynamics."
              />
              <FeatureCard
                icon=">"
                title="Derived Light"
                description="Light is derived from partition operations (c = Δx/τ). The same physics that creates the molecules creates the light that observes them."
              />
              <FeatureCard
                icon="}"
                title="Ray-Marched Observation"
                description="A volumetric ray march through the gas/fluid computes optical, kinetic, and thermodynamic observables simultaneously at each step."
              />
            </div>
          </Layout>
        </section>

        {/* ── Modes ── */}
        <section className="py-24 border-t border-gray-800/50">
          <Layout>
            <div className="grid grid-cols-2 gap-12 md:grid-cols-1">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
                viewport={{ once: true }}
                className="bg-gradient-to-br from-blue-900/20 to-dark border border-blue-800/30 rounded-2xl p-8"
              >
                <div className="text-blue-400 text-sm font-mono uppercase tracking-wider mb-3">Gas Dynamics</div>
                <h3 className="text-2xl font-bold text-light mb-4">Kinetic Theory</h3>
                <ul className="text-gray-400 text-sm space-y-2">
                  <li>Ideal gas law PV = NkBT (categorical balance)</li>
                  <li>Maxwell-Boltzmann distribution (bounded at c)</li>
                  <li>Equipartition U = 3/2 NkBT</li>
                  <li>Adiabatic processes PV^γ = const</li>
                  <li>Mean free path and collision statistics</li>
                </ul>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
                viewport={{ once: true }}
                className="bg-gradient-to-br from-orange-900/20 to-dark border border-orange-800/30 rounded-2xl p-8"
              >
                <div className="text-orange-400 text-sm font-mono uppercase tracking-wider mb-3">Fluid Dynamics</div>
                <h3 className="text-2xl font-bold text-light mb-4">Navier-Stokes</h3>
                <ul className="text-gray-400 text-sm space-y-2">
                  <li>Viscosity μ = τ_c × g (2.9% error, 12 liquids)</li>
                  <li>Navier-Stokes from Kirchhoff on partition networks</li>
                  <li>Poiseuille flow recovery (0.9% error)</li>
                  <li>Stokes-Einstein diffusion</li>
                  <li>Gas → liquid phase transitions via network density</li>
                </ul>
              </motion.div>
            </div>
          </Layout>
        </section>

        {/* ── CTA ── */}
        <section className="py-24 border-t border-gray-800/50">
          <Layout>
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">Ready to explore?</h2>
              <p className="text-gray-400 mb-8 max-w-md mx-auto">
                No installation. No account. Just open the simulator
                and watch spectral-native dynamics in real time.
              </p>
              <Link
                href="/simulate"
                className="inline-block px-10 py-4 bg-primaryDark text-dark font-bold rounded-lg
                           hover:bg-primaryDark/80 transition-colors text-lg"
              >
                Launch Simulator
              </Link>
            </div>
          </Layout>
        </section>
      </div>
    </>
  );
}
