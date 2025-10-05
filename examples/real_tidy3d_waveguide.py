#!/usr/bin/env python3
"""
REAL Tidy3D Waveguide Simulation
=================================

This runs an ACTUAL FDTD simulation using Tidy3D's cloud solver.
Based on Tidy3D tutorial notebooks from:
https://github.com/flexcompute/tidy3d-notebooks

NO MOCKING - This is real electromagnetic simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tidy3d as td
    from tidy3d import web
    print(f"✓ Tidy3D version: {td.__version__}")
    TIDY3D_AVAILABLE = True
except ImportError as e:
    print(f"✗ Tidy3D not available: {e}")
    TIDY3D_AVAILABLE = False
    sys.exit(1)


def create_silicon_waveguide_simulation():
    """
    Create a real silicon strip waveguide simulation.

    This is based on standard silicon photonics parameters:
    - 500nm x 220nm Si waveguide
    - 1.55 μm wavelength (C-band)
    - SiO2 cladding
    """

    print("\n" + "="*70)
    print("REAL TIDY3D FDTD SIMULATION")
    print("="*70)
    print()

    # Physical parameters
    wavelength = 1.55  # μm
    freq0 = td.C_0 / wavelength  # Hz
    freqw = freq0 / 10  # Frequency width

    # Waveguide geometry
    waveguide_width = 0.5  # μm
    waveguide_height = 0.22  # μm
    waveguide_length = 10.0  # μm

    # Materials (real refractive indices)
    n_si = 3.48
    n_sio2 = 1.45

    silicon = td.Medium(permittivity=n_si**2, name="Si")
    oxide = td.Medium(permittivity=n_sio2**2, name="SiO2")

    print(f"Wavelength: {wavelength} μm")
    print(f"Frequency: {freq0/1e14:.3f} × 10^14 Hz")
    print(f"Waveguide: {waveguide_width} × {waveguide_height} μm")
    print(f"Length: {waveguide_length} μm")
    print()

    # Simulation domain size
    sim_size = [waveguide_length + 4, 4, 3]  # Add padding

    # Create waveguide structure
    waveguide = td.Structure(
        geometry=td.Box(
            center=[0, 0, waveguide_height/2],
            size=[td.inf, waveguide_width, waveguide_height]
        ),
        medium=silicon,
        name="waveguide"
    )

    # Substrate
    substrate = td.Structure(
        geometry=td.Box(
            center=[0, 0, -1],
            size=[td.inf, td.inf, 2]
        ),
        medium=oxide,
        name="substrate"
    )

    # Mode source (inject TE mode)
    mode_spec = td.ModeSpec(
        num_modes=1,
        target_neff=n_si,  # Target effective index
    )

    source = td.ModeSource(
        center=[-waveguide_length/2 + 1, 0, waveguide_height/2],
        size=[0, 3, 2],
        source_time=td.GaussianPulse(
            freq0=freq0,
            fwidth=freqw
        ),
        mode_spec=mode_spec,
        direction="+",
        name="mode_source"
    )

    # Mode monitor (measure transmission)
    monitor = td.ModeMonitor(
        center=[waveguide_length/2 - 1, 0, waveguide_height/2],
        size=[0, 3, 2],
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_monitor"
    )

    # Field monitor (visualize fields)
    field_monitor = td.FieldMonitor(
        center=[0, 0, waveguide_height/2],
        size=[waveguide_length, 3, 0],
        freqs=[freq0],
        name="field_monitor"
    )

    # Create simulation
    simulation = td.Simulation(
        center=[0, 0, 0],
        size=sim_size,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=20,  # 20 points per wavelength
            wavelength=wavelength
        ),
        structures=[substrate, waveguide],
        sources=[source],
        monitors=[monitor, field_monitor],
        run_time=40/freqw,  # Run for 40 periods
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        medium=oxide,  # Background medium
        symmetry=(0, 0, 0),  # No symmetry (for simplicity)
    )

    print("✓ Simulation created")
    print(f"  Grid points: ~{np.prod(simulation.num_cells)/1e6:.1f}M cells")
    print(f"  Run time: {simulation.run_time*1e12:.1f} ps")
    print()

    return simulation


def run_simulation(simulation, task_name="silicon_waveguide"):
    """
    Run the simulation on Tidy3D cloud.

    NOTE: This requires a Tidy3D account and API key.
    """

    print("Submitting to Tidy3D cloud...")
    print("(This requires API key - configure with: tidy3d configure)")
    print()

    try:
        # Upload and run
        job = web.Job(simulation=simulation, task_name=task_name)

        print(f"Job created: {job.task_id}")
        print("Estimating cost...")

        # Estimate flex credits needed
        job.estimate_cost()

        print("\nStarting simulation...")
        job.run(path="simulation_data.hdf5")

        print("✓ Simulation complete!")

        # Load results
        sim_data = job.load_results()

        return sim_data

    except Exception as e:
        print(f"\n⚠ Simulation submission failed: {e}")
        print("\nThis could be because:")
        print("  1. No Tidy3D API key configured")
        print("  2. No internet connection")
        print("  3. Insufficient flex credits")
        print("\nTo configure: Run 'tidy3d configure' in terminal")
        return None


def visualize_results(sim_data, simulation):
    """
    Visualize REAL simulation results.
    """

    if sim_data is None:
        print("\n⚠ No simulation data to visualize")
        return

    print("\n" + "="*70)
    print("ANALYZING RESULTS")
    print("="*70)
    print()

    # Extract transmission
    mode_amps = sim_data["mode_monitor"].amps
    transmission = np.abs(mode_amps.sel(direction="+", mode_index=0).values[0])**2

    print(f"Transmission: {transmission*100:.2f}%")

    # Get field data
    field_data = sim_data["field_monitor"]
    Ex = field_data.Ex.values[0]  # First frequency

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0a0a0a')

    # Plot 1: Simulation structure
    ax1 = axes[0, 0]
    ax1.set_facecolor('#0f0f0f')
    simulation.plot(z=0.11, ax=ax1)
    ax1.set_title('Waveguide Structure (Top View)', color='cyan', fontsize=14)

    # Plot 2: Electric field
    ax2 = axes[0, 1]
    ax2.set_facecolor('#0f0f0f')
    im = ax2.imshow(np.abs(Ex).T, cmap='hot', aspect='auto',
                     extent=[-5, 5, -1.5, 1.5])
    ax2.set_title('|Ex| Field Distribution', color='cyan', fontsize=14)
    ax2.set_xlabel('Propagation (μm)', color='white')
    ax2.set_ylabel('Width (μm)', color='white')
    plt.colorbar(im, ax=ax2, label='Field amplitude')

    # Plot 3: Mode profile
    ax3 = axes[1, 0]
    ax3.set_facecolor('#0f0f0f')
    ax3.plot(field_data.y, np.abs(Ex[len(Ex)//2, :]), 'cyan', linewidth=2)
    ax3.set_title('Mode Profile (Cross-section)', color='cyan', fontsize=14)
    ax3.set_xlabel('Width (μm)', color='white')
    ax3.set_ylabel('|Ex|', color='white')
    ax3.grid(True, alpha=0.2)

    # Plot 4: Transmission summary
    ax4 = axes[1, 1]
    ax4.set_facecolor('#0f0f0f')
    ax4.axis('off')

    results_text = f"""
REAL TIDY3D FDTD RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRANSMISSION:  {transmission*100:.2f}%
LOSS:          {-10*np.log10(transmission):.2f} dB

SIMULATION PARAMETERS:
Grid cells:    {np.prod(simulation.num_cells)/1e6:.2f}M
Run time:      {simulation.run_time*1e12:.1f} ps
Wavelength:    1.55 μm
Mode:          TE fundamental

WAVEGUIDE:
Width:         0.5 μm
Height:        0.22 μm
Material:      Silicon (n=3.48)
Cladding:      SiO2 (n=1.45)

✓ This is REAL FDTD data
✓ Solved on Tidy3D cloud
✓ Full Maxwell equations
    """

    ax4.text(0.1, 0.95, results_text, transform=ax4.transAxes,
            fontsize=10, color='#4ade80', family='monospace',
            verticalalignment='top')

    fig.suptitle('REAL TIDY3D FDTD SIMULATION RESULTS',
                 fontsize=18, color='cyan', fontweight='bold', y=0.98)

    output_file = "docs/demos/REAL_tidy3d_simulation.png"
    plt.savefig(output_file, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()

    print(f"\n✓ Results saved: {output_file}")
    print()

    return transmission


def main():
    """Run real Tidy3D simulation."""

    if not TIDY3D_AVAILABLE:
        print("✗ Tidy3D not installed")
        print("  Install with: pip install tidy3d")
        return

    # Create simulation
    simulation = create_silicon_waveguide_simulation()

    # Visualize structure before running
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='#0a0a0a')
    ax.set_facecolor('#0f0f0f')
    simulation.plot(z=0.11, ax=ax)
    ax.set_title('Silicon Waveguide FDTD Simulation Setup',
                 color='cyan', fontsize=16, fontweight='bold')
    plt.savefig("docs/demos/simulation_setup.png", dpi=150,
                facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    print("✓ Simulation setup saved")

    # Ask user before submitting
    print("\n" + "="*70)
    print("READY TO SUBMIT TO TIDY3D CLOUD")
    print("="*70)
    print("\nThis will:")
    print("  1. Upload simulation to Tidy3D servers")
    print("  2. Run FDTD solver on cloud hardware")
    print("  3. Use flex credits from your account")
    print()

    response = input("Continue with cloud simulation? (yes/no): ")

    if response.lower() == 'yes':
        # Run real simulation
        sim_data = run_simulation(simulation)

        # Visualize results
        if sim_data:
            transmission = visualize_results(sim_data, simulation)
            print(f"\n✅ REAL SIMULATION COMPLETE!")
            print(f"   Transmission: {transmission*100:.2f}%")
    else:
        print("\n⚠ Simulation cancelled by user")
        print("  Setup visualization saved to: docs/demos/simulation_setup.png")

    print()


if __name__ == "__main__":
    main()
