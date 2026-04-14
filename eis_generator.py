"""
Synthetic EIS Spectra Generator
Models battery impedance using Randles circuit (Rs + Rct || Warburg element)
Impedance parameters drift with cycle aging, creating a physical EIS "fingerprint"
"""

import numpy as np
import json
from scipy import signal
import matplotlib.pyplot as plt

class EISSpectraGenerator:
    """Generate synthetic EIS spectra that realistically degrade with battery age."""
    
    def __init__(self, ground_truth_file='battery_ground_truth.json'):
        """
        Args:
            ground_truth_file: JSON file with internal variables from PyBaMM sim
        """
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)
        
        self.eis_data = []
        print(f"📊 Loaded ground truth from {ground_truth_file}")
    
    def _randles_circuit(self, frequency, Rs, Rct, sigma_warburg):
        """
        Randles circuit impedance model.
        Z = Rs + Rct / (1 + sqrt(-j*omega*tau_w))
        where tau_w = 1/sigma_warburg^2, and Warburg element ~1/sqrt(f)
        
        Args:
            frequency: array of frequencies (Hz)
            Rs: Ohmic/solution resistance (Ω)
            Rct: Charge-transfer resistance (Ω)
            sigma_warburg: Warburg coefficient (Ω*Hz^-0.5)
        
        Returns:
            Z: Complex impedance
        """
        omega = 2 * np.pi * frequency
        
        # Warburg impedance: Z_w = sigma / sqrt(j*omega)
        # where j*omega = (j*omega + (-j*omega))/2 ≈ for small omega, use (1-j)/sqrt(omega)
        z_warburg = sigma_warburg / np.sqrt(omega + 1e-12) * (1 - 1j) / np.sqrt(2)
        
        # Parallel: Rct || Z_w
        z_parallel = (Rct * z_warburg) / (Rct + z_warburg)
        
        # Series: Rs + parallel
        z_total = Rs + z_parallel
        
        return z_total
    
    def generate_eis_spectra(self, frequency_range=(1e-2, 1e5), n_freq=100):
        """
        Generate synthetic EIS spectra for each cycle.
        
        Args:
            frequency_range: (min_freq, max_freq) in Hz
            n_freq: Number of frequency points (logarithmic spacing)
        
        Returns:
            EIS features for anomaly detection
        """
        freq = np.logspace(np.log10(frequency_range[0]), 
                          np.log10(frequency_range[1]), 
                          n_freq)
        
        print(f"🌊 Generating EIS spectra across {len(self.ground_truth)} cycles...")
        print(f"   Frequency range: {frequency_range[0]:.2e} - {frequency_range[1]:.2e} Hz")
        
        for idx, gt in enumerate(self.ground_truth):
            cycle = gt['cycle']
            
            # EIS parameters drift with aging (calibrated to realistic values)
            # Ohmic resistance: slight increase with SEI
            Rs = (gt['ohmic_resistance_mOhm'] / 1000) + np.random.normal(0, 0.0005)
            
            # Charge-transfer resistance: increases with SEI thickness exponentially
            Rct = (gt['ct_resistance_mOhm'] / 1000) + np.random.normal(0, 0.001)
            
            # Warburg coefficient: increases with SEI and surface irregularity
            sei_um = gt['sei_thickness_um']
            sigma_warburg = 0.01 + 0.002 * sei_um + np.random.normal(0, 0.0002)
            
            # Generate impedance
            Z = self._randles_circuit(freq, Rs, Rct, sigma_warburg)
            
            # Nyquist plot data
            Z_real = Z.real
            Z_imag = -Z.imag  # Convention: plot -Im(Z)
            
            # Extract arc diameter (charge-transfer arc)
            arc_diameter = np.max(Z_imag) * 2
            
            # Peak frequency (where -Im(Z) is maximum for charge-transfer arc)
            peak_idx = np.argmax(Z_imag)
            peak_freq = freq[peak_idx]
            
            # Store spectrum and extracted features
            spectrum = {
                'cycle': cycle,
                'frequency_Hz': freq.tolist(),
                'Z_real_Ohm': Z_real.tolist(),
                'Z_imag_Ohm': Z_imag.tolist(),
                'Rs_Ohm': Rs,
                'Rct_Ohm': Rct,
                'sigma_warburg_Ohm_Hz_neg05': sigma_warburg,
                'arc_diameter_Ohm': arc_diameter,
                'peak_freq_Hz': peak_freq,
                'sei_thickness_um': sei_um,
            }
            
            self.eis_data.append(spectrum)
            
            if (cycle + 1) % 100 == 0:
                print(f"  ✓ Cycle {cycle+1}: Rs={Rs*1000:.3f}mΩ, "
                      f"Rct={Rct*1000:.3f}mΩ, σ_w={sigma_warburg:.5f}Ω·Hz⁻⁰·⁵")
        
        print("✅ EIS generation complete!")
        return self.eis_data
    
    def extract_features(self):
        """Extract machine-learning features from EIS spectra."""
        features = []
        
        for spectrum in self.eis_data:
            feat = {
                'cycle': spectrum['cycle'],
                'Rs': spectrum['Rs_Ohm'],
                'Rct': spectrum['Rct_Ohm'],
                'sigma_warburg': spectrum['sigma_warburg_Ohm_Hz_neg05'],
                'arc_diameter': spectrum['arc_diameter_Ohm'],
                'peak_freq': spectrum['peak_freq_Hz'],
                'sei_thickness_um': spectrum['sei_thickness_um'],
                # Derived features for better anomaly detection
                'Rs_normalized': spectrum['Rs_Ohm'] / 0.01,  # Relative to initial
                'Rct_normalized': spectrum['Rct_Ohm'] / 0.05,
                'Rs_change_rate': spectrum['Rs_Ohm'] * (spectrum['cycle'] + 1),
                'Rct_change_rate': spectrum['Rct_Ohm'] * (spectrum['cycle'] + 1),
            }
            features.append(feat)
        
        return features
    
    def export_eis_data(self, filename='eis_spectra.json'):
        """Export full EIS spectra."""
        with open(filename, 'w') as f:
            json.dump(self.eis_data, f)
        print(f"💾 EIS spectra exported to {filename}")
    
    def plot_nyquist(self, cycle_indices=[0, 100, 200, 300, 400], save_fig='nyquist_evolution.png'):
        """Plot Nyquist plots at selected cycle checkpoints."""
        n_plots = len(cycle_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        for ax_idx, cycle_idx in enumerate(cycle_indices):
            spectrum = self.eis_data[cycle_idx]
            
            ax = axes[ax_idx]
            ax.plot(spectrum['Z_real_Ohm'], spectrum['Z_imag_Ohm'], 'b-', linewidth=2, label='Impedance')
            
            # Mark features
            ax.scatter([spectrum['Rs_Ohm']], [0], color='red', s=100, zorder=5, label=f"Rs={spectrum['Rs_Ohm']*1000:.2f}mΩ")
            
            ax.set_xlabel('Z_real (Ω)', fontsize=11)
            ax.set_ylabel('-Z_imag (Ω)', fontsize=11)
            ax.set_title(f"Cycle {spectrum['cycle']}\nSEI={spectrum['sei_thickness_um']:.3f}μm", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"📈 Nyquist plots saved to {save_fig}")
        plt.close()


if __name__ == "__main__":
    # Generate EIS spectra
    gen = EISSpectraGenerator('battery_ground_truth.json')
    gen.generate_eis_spectra()
    gen.export_eis_data()
    gen.plot_nyquist()
    
    # Extract ML features
    features = gen.extract_features()
    print(f"\n📊 Extracted {len(features)} feature vectors")
    print(f"   Feature keys: {list(features[0].keys())}")
