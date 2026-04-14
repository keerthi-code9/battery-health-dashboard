"""
Degradation Stage Labeler
Assigns stages 1-5 based on PyBaMM internal variables (SEI, Li plating, capacity fade)
These serve as ground-truth labels that would come from NMR in production.
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DegradationStage:
    """Represents a degradation stage threshold."""
    stage: int
    name: str
    sei_threshold_um: float
    li_plating_threshold_um: float
    capacity_fade_threshold_pct: float
    description: str


class DegradationStageLabelizer:
    """
    Assigns degradation stages based on physical degradation markers.
    
    In production:
      - These labels come from NMR (Nuclear Magnetic Resonance) data
      - NMR can quantify SEI layer growth and detect lithium plating
    
    In our prototype:
      - We use PyBaMM's internal state variables (SEI, Li plating)
      - This is defensible as a physics-based ground truth oracle
      - We explicitly frame it as "PyBaMM-simulated NMR equivalent"
    """
    
    # Define degradation stages with realistic thresholds
    STAGES = [
        DegradationStage(
            stage=1,
            name="Pristine",
            sei_threshold_um=0.0,
            li_plating_threshold_um=0.0,
            capacity_fade_threshold_pct=0.0,
            description="New battery, no significant degradation"
        ),
        DegradationStage(
            stage=2,
            name="Early Degradation",
            sei_threshold_um=0.5,
            li_plating_threshold_um=0.0,
            capacity_fade_threshold_pct=2.0,
            description="SEI layer starting to grow, minimal li plating"
        ),
        DegradationStage(
            stage=3,
            name="Moderate Degradation",
            sei_threshold_um=1.5,
            li_plating_threshold_um=0.1,
            capacity_fade_threshold_pct=5.0,
            description="Significant SEI growth, early li plating detection"
        ),
        DegradationStage(
            stage=4,
            name="Advanced Degradation",
            sei_threshold_um=3.0,
            li_plating_threshold_um=0.5,
            capacity_fade_threshold_pct=10.0,
            description="Heavy SEI, substantial li plating, high risk"
        ),
        DegradationStage(
            stage=5,
            name="Critical",
            sei_threshold_um=5.0,
            li_plating_threshold_um=1.0,
            capacity_fade_threshold_pct=15.0,
            description="Severe degradation, immediate maintenance needed"
        ),
    ]
    
    def __init__(self, ground_truth_file='battery_ground_truth.json'):
        """
        Args:
            ground_truth_file: JSON with PyBaMM internal variables
        """
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)
        
        self.labels = []
        print("🏷️  Degradation stage labeler initialized")
        print("   Using PyBaMM internal variables as quantum-proxy ground truth")
    
    def assign_stages(self) -> List[Dict]:
        """
        Assign stages to each cycle based on physical markers.
        
        Logic: A cycle enters stage N when ANY degradation metric exceeds that stage's threshold.
        This is conservative: we want early warning, not late detection.
        """
        print(f"🔬 Assigning degradation stages to {len(self.ground_truth)} cycles...")
        
        stage_transitions = {}  # Track when each stage is first reached
        
        for gt in self.ground_truth:
            cycle = gt['cycle']
            sei = gt['sei_thickness_um']
            li_plating = gt['li_plating_um']
            capacity_fade = gt['capacity_fade_pct']
            
            # Determine stage: find the highest stage threshold exceeded
            assigned_stage = 1  # Default to pristine
            
            for stage_def in self.STAGES[1:]:  # Skip pristine stage
                # ANY metric exceeding threshold → enter that stage
                if (sei >= stage_def.sei_threshold_um or 
                    li_plating >= stage_def.li_plating_threshold_um or
                    capacity_fade >= stage_def.capacity_fade_threshold_pct):
                    assigned_stage = stage_def.stage
                
                # Record stage transition
                if assigned_stage not in stage_transitions and assigned_stage == stage_def.stage:
                    stage_transitions[assigned_stage] = cycle
            
            label = {
                'cycle': cycle,
                'stage': assigned_stage,
                'stage_name': self.STAGES[assigned_stage - 1].name,
                'sei_thickness_um': sei,
                'li_plating_um': li_plating,
                'capacity_fade_pct': capacity_fade,
                'ohmic_resistance_mOhm': gt['ohmic_resistance_mOhm'],
                'ct_resistance_mOhm': gt['ct_resistance_mOhm'],
            }
            
            self.labels.append(label)
        
        print("✅ Stage assignment complete!")
        print("\n🎯 Stage Transition Timeline:")
        for stage in sorted(stage_transitions.keys()):
            cycle = stage_transitions[stage]
            stage_def = self.STAGES[stage - 1]
            print(f"   Stage {stage} ({stage_def.name}): Cycle {cycle}")
        
        return self.labels
    
    def get_stage_distribution(self):
        """Get distribution of stages across cycles."""
        from collections import Counter
        stages = [l['stage'] for l in self.labels]
        dist = Counter(stages)
        print("\n📊 Stage Distribution:")
        for stage in sorted(dist.keys()):
            count = dist[stage]
            stage_def = self.STAGES[stage - 1]
            pct = 100 * count / len(self.labels)
            print(f"   Stage {stage} ({stage_def.name:20s}): {count:3d} cycles ({pct:5.1f}%)")
        return dist
    
    def export_labels(self, filename='degradation_labels.json'):
        """Export stage labels for ML training."""
        with open(filename, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"💾 Labels exported to {filename}")
    
    def get_stage_thresholds(self) -> Dict:
        """Return thresholds for reference."""
        thresholds = {}
        for stage_def in self.STAGES:
            thresholds[stage_def.stage] = {
                'name': stage_def.name,
                'sei_um': stage_def.sei_threshold_um,
                'li_plating_um': stage_def.li_plating_threshold_um,
                'capacity_fade_pct': stage_def.capacity_fade_threshold_pct,
            }
        return thresholds


if __name__ == "__main__":
    # Assign stages
    labeler = DegradationStageLabelizer('battery_ground_truth.json')
    labels = labeler.assign_stages()
    labeler.get_stage_distribution()
    labeler.export_labels()
    
    print("\n🔍 Stage Definitions:")
    for stage_def in labeler.STAGES:
        print(f"\n   Stage {stage_def.stage}: {stage_def.name}")
        print(f"   └─ {stage_def.description}")
        print(f"      Thresholds: SEI ≥ {stage_def.sei_threshold_um}μm, "
              f"Li-plating ≥ {stage_def.li_plating_threshold_um}μm, "
              f"Capacity fade ≥ {stage_def.capacity_fade_threshold_pct}%")
