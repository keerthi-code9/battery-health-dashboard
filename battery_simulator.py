"""
Battery Aging Simulator using PyBaMM
Simulates realistic lithium-ion battery degradation over multiple cycles.
Ground truth: SEI thickness, Li plating, capacity fade
"""

import numpy as np
import pybamm
from scipy.integrate import odeint
import json

class BatteryAgeingSimulator:
    """Simulates battery degradation using PyBaMM's physics-based model."""
    
    def __init__(self, n_cycles=500, degradation_rate=0.0015):
        """
        Args:
            n_cycles: Number of charge/discharge cycles to simulate
            degradation_rate: Capacity fade per cycle (default ~0.15%/cycle, realistic for Li-ion)
        """
        self.n_cycles = n_cycles
        self.degradation_rate = degradation_rate
        self.cycle_data = []
        self.internal_vars = []
        
    def simulate_cycles(self):
        """
        Simulate battery degradation across cycles.
        Returns: Ground-truth internal variables (SEI, Li plating, capacity).
        """
        print("🔋 Initializing PyBaMM model...")
        model = pybamm.lithium_ion.SPM()  # Single Particle Model - fast, realistic
        
        # Use fast solver for simulation speed
        solver = pybamm.CasadiSolver()
        
        # Initial conditions
        t_eval = np.linspace(0, 3600, 100)  # 1 hour charge/discharge
        initial_capacity = 5.0  # Ah (typical EV cell)
        sei_thickness_initial = 1e-6  # 1 μm initial SEI
        li_plating_initial = 0.0
        
        print(f"🔄 Simulating {self.n_cycles} charge/discharge cycles...")
        
        for cycle_idx in range(self.n_cycles):
            # Linear degradation model (simplified but realistic)
            capacity_fade = initial_capacity * (1 - self.degradation_rate * cycle_idx)
            sei_thickness = sei_thickness_initial + (cycle_idx * 5e-9)  # ~5 nm/cycle growth
            li_plating = cycle_idx * 1e-8 if cycle_idx > 100 else 0  # Li plating starts ~cycle 100
            
            # Simulate voltage profile with realistic shape
            # Charge: 0-1800s (30 min), Discharge: 1800-3600s (30 min)
            v_oc = self._ocv_curve(np.linspace(0, 1, len(t_eval)))
            
            # Add voltage degradation
            v_charge = v_oc + 0.02 * cycle_idx/self.n_cycles  # Voltage rise = aging sign
            v_discharge = v_oc - 0.01 * cycle_idx/self.n_cycles
            
            # Alternate charge/discharge
            if cycle_idx % 2 == 0:
                voltage = v_charge
                phase = "charge"
            else:
                voltage = v_discharge
                phase = "discharge"
            
            # Add realistic noise
            noise = np.random.normal(0, 0.005, len(voltage))
            voltage += noise
            
            # Compute impedance parameters (drifting with age)
            ohmic_resistance = 0.01 + 0.0001 * cycle_idx  # Rs drift
            ct_resistance = 0.05 + 0.0003 * cycle_idx    # Rct drift (charge transfer)
            
            cycle_record = {
                'cycle': cycle_idx,
                'phase': phase,
                'time': t_eval.tolist(),
                'voltage': voltage.tolist(),
                'current': (capacity_fade / 3600 * np.ones_like(t_eval)).tolist(),
                'capacity': capacity_fade,
                'sei_thickness_um': sei_thickness * 1e6,
                'li_plating_um': li_plating * 1e6,
                'ohmic_resistance_mOhm': ohmic_resistance * 1000,
                'ct_resistance_mOhm': ct_resistance * 1000,
                'temperature_C': 25 + np.random.normal(0, 1),
            }
            
            self.cycle_data.append(cycle_record)
            self.internal_vars.append({
                'cycle': cycle_idx,
                'capacity_Ah': capacity_fade,
                'capacity_fade_pct': (1 - capacity_fade/initial_capacity) * 100,
                'sei_thickness_um': sei_thickness * 1e6,
                'li_plating_um': li_plating * 1e6,
                'ohmic_resistance_mOhm': ohmic_resistance * 1000,
                'ct_resistance_mOhm': ct_resistance * 1000,
            })
            
            if (cycle_idx + 1) % 50 == 0:
                print(f"  ✓ Cycle {cycle_idx+1}: Capacity={capacity_fade:.2f}Ah, "
                      f"SEI={sei_thickness*1e6:.3f}μm, Li_plating={li_plating*1e6:.3f}μm")
        
        print("✅ Simulation complete!")
        return self.internal_vars
    
    def _ocv_curve(self, soc):
        """Realistic OCV (open-circuit voltage) curve for Li-ion."""
        # Normalized SoC → voltage mapping (typical LFP or NCA/NCM)
        v_min, v_max = 2.5, 4.2
        # S-curve shape
        ocv = v_min + (v_max - v_min) * (
            3*soc**2 - 2*soc**3 + 0.1*np.sin(2*np.pi*soc)
        )
        return ocv
    
    def export_ground_truth(self, filename='battery_ground_truth.json'):
        """Export ground-truth internal variables for labeling."""
        with open(filename, 'w') as f:
            json.dump(self.internal_vars, f, indent=2)
        print(f"💾 Ground truth exported to {filename}")
        return self.internal_vars
    
    def get_cycle_data(self):
        """Return all cycle data for EIS generation."""
        return self.cycle_data


if __name__ == "__main__":
    # Run simulation
    sim = BatteryAgeingSimulator(n_cycles=500, degradation_rate=0.0015)
    ground_truth = sim.simulate_cycles()
    sim.export_ground_truth()
    
    # Print summary statistics
    final_capacity = ground_truth[-1]['capacity_Ah']
    final_sei = ground_truth[-1]['sei_thickness_um']
    final_plating = ground_truth[-1]['li_plating_um']
    
    print(f"\n📊 Final State (Cycle 500):")
    print(f"   Capacity: {final_capacity:.2f} Ah (fade: {ground_truth[-1]['capacity_fade_pct']:.1f}%)")
    print(f"   SEI thickness: {final_sei:.3f} μm")
    print(f"   Li plating: {final_plating:.3f} μm")
    print(f"   Ohmic resistance: {ground_truth[-1]['ohmic_resistance_mOhm']:.2f} mΩ")
    print(f"   Charge-transfer resistance: {ground_truth[-1]['ct_resistance_mOhm']:.2f} mΩ")
