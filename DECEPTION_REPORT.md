================================================================================
🚨 DECEPTION DETECTOR REPORT
================================================================================

Total deceptions found: 25
  CRITICAL: 4
  HIGH: 21

================================================================================
🔴 CRITICAL DECEPTIONS (Actively Misleading)
================================================================================

📍 agents/specialized/fdtd_agent.py:73
   Type: MISLEADING_NAME
   Pattern: Function "_simulate_waveguide" claims to simulate but returns dict
   Code: def _simulate_waveguide(...)

📍 agents/specialized/fdtd_agent.py:164
   Type: MISLEADING_NAME
   Pattern: Function "_simulate_ring_resonator" claims to simulate but returns dict
   Code: def _simulate_ring_resonator(...)

📍 agents/specialized/fdtd_agent.py:183
   Type: MISLEADING_NAME
   Pattern: Function "_simulate_mzi" claims to simulate but returns dict
   Code: def _simulate_mzi(...)

📍 agents/specialized/fdtd_agent.py:203
   Type: MISLEADING_NAME
   Pattern: Function "_simulate_photonic_crystal" claims to simulate but returns dict
   Code: def _simulate_photonic_crystal(...)

================================================================================
⚠️  HIGH SEVERITY DECEPTIONS
================================================================================


📁 agents/engineering_agent.py
   Found 1 deceptive patterns:
   Line 70: Sine/cosine fake waveform

📁 agents/specialized/fdtd_agent.py
   Found 1 deceptive patterns:
   Line 190: Sine/cosine fake waveform

📁 deception_detector.py
   Found 3 deceptive patterns:
   Line 27: Mock/Fake comment
   Line 28: MEEP fallback to analytical
   Line 32: Analytical approximation function

📁 demos/impressive_workflow.py
   Found 7 deceptive patterns:
   Line 136: Mock/Fake comment
   Line 164: Gaussian approximation
   Line 188: Mock/Fake comment
   Line 194: Gaussian approximation
   Line 256: Mock/Fake comment
   ... and 2 more

📁 examples/meep_waveguide_fdtd.py
   Found 3 deceptive patterns:
   Line 26: MEEP fallback to analytical
   Line 148: Analytical approximation function
   Line 312: Gaussian approximation

📁 examples/tidy3d_waveguide_transmission.py
   Found 2 deceptive patterns:
   Line 130: Gaussian approximation
   Line 286: Gaussian approximation

📁 visualizations/renderer.py
   Found 4 deceptive patterns:
   Line 125: Random data generation
   Line 212: Random data generation
   Line 295: Random data generation
   Line 346: Random data generation

================================================================================
RECOMMENDATIONS
================================================================================

1. Remove or clearly label all analytical approximations
2. Rename misleading function names (e.g., 'simulate' → 'approximate')
3. Add docstrings explaining what's real vs. mock
4. Replace fake data with real simulations or remove features
5. Be explicit about fallback behaviors
