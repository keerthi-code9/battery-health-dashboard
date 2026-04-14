"""
Anomaly Detection Model Training
Trains Isolation Forest and LSTM to detect battery degradation anomalies.
Compares performance vs. classical voltage-threshold approach.
"""

import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

class AnomalyDetectionEngine:
    """Train and evaluate anomaly detection models for battery health."""
    
    def __init__(self, eis_features_file='eis_features.json', 
                 labels_file='degradation_labels.json'):
        """Load EIS features and stage labels."""
        with open(eis_features_file, 'r') as f:
            self.eis_features = json.load(f)
        
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.anomaly_scores = []
        
        print("🤖 Anomaly Detection Engine initialized")
    
    def _prepare_feature_matrix(self):
        """Convert EIS features to ML-ready format."""
        feature_keys = ['Rs', 'Rct', 'sigma_warburg', 'arc_diameter', 
                       'peak_freq', 'Rs_normalized', 'Rct_normalized']
        
        X = np.array([[f[key] for key in feature_keys] for f in self.eis_features])
        y = np.array([l['stage'] for l in self.labels])
        
        return X, y, feature_keys
    
    def train_isolation_forest(self, contamination=0.1):
        """
        Train Isolation Forest for anomaly detection.
        
        Anomalies are defined as cycles deviating from "pristine" EIS signature.
        As battery ages, Rs, Rct, and Warburg coefficient all increase → anomaly.
        
        Args:
            contamination: Expected fraction of anomalies (aggressive: 10%)
        """
        print(f"🌲 Training Isolation Forest (contamination={contamination})...")
        
        X, y, feature_keys = self._prepare_feature_matrix()
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Get anomaly predictions (-1 = anomaly, 1 = normal)
        # and anomaly scores (lower = more anomalous)
        anomaly_preds = self.isolation_forest.fit_predict(X_scaled)
        anomaly_scores = self.isolation_forest.score_samples(X_scaled)
        
        # Convert to 0-1 scale: higher = more anomalous
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        normalized_scores = (max_score - anomaly_scores) / (max_score - min_score)
        
        self.anomaly_scores = normalized_scores
        
        # Analyze detection performance
        self._analyze_detection_performance(y, normalized_scores)
        
        return self.isolation_forest, normalized_scores
    
    def _analyze_detection_performance(self, y_true, y_score):
        """Analyze anomaly detection vs stage transitions."""
        print("\n📊 Detection Performance Analysis:")
        
        # Find stage transitions
        stage_transitions = {}
        for i, label in enumerate(self.labels):
            if i > 0 and label['stage'] > self.labels[i-1]['stage']:
                stage_transitions[label['stage']] = i
        
        # Find when anomaly score first exceeds threshold (0.5)
        threshold = 0.5
        anomaly_detected_cycle = None
        for i, score in enumerate(y_score):
            if score > threshold:
                anomaly_detected_cycle = i
                break
        
        # Find EOL by voltage threshold (classical BMS)
        # Assume 80% capacity fade triggers alarm
        eol_cycle_voltage = None
        for label in self.labels:
            if label['capacity_fade_pct'] >= 20:  # 80% of original capacity
                eol_cycle_voltage = label['cycle']
                break
        
        if anomaly_detected_cycle is not None:
            early_warning = (eol_cycle_voltage - anomaly_detected_cycle) if eol_cycle_voltage else 0
            print(f"   Anomaly detected at cycle {anomaly_detected_cycle}")
            if eol_cycle_voltage:
                print(f"   Classical BMS (80% capacity) at cycle {eol_cycle_voltage}")
                print(f"   ✅ Early warning: {early_warning} cycles ({100*early_warning/eol_cycle_voltage:.0f}% earlier)")
        
        # Calculate AUC
        # Generate binary labels: 1 = stage > 2 (degraded)
        y_degraded = (y_true > 2).astype(int)
        if len(np.unique(y_degraded)) > 1:
            auc = roc_auc_score(y_degraded, y_score)
            print(f"   ROC-AUC (Stage > 2): {auc:.3f}")
        
        return anomaly_detected_cycle, eol_cycle_voltage
    
    def train_lstm_sequential(self, lookback=20):
        """
        Train LSTM for sequential anomaly detection.
        Uses EIS feature time-series to predict next-cycle degradation.
        
        Args:
            lookback: Number of cycles to look back
        """
        try:
            from keras.models import Sequential
            from keras.layers import LSTM, Dense, Dropout
            from keras.optimizers import Adam
        except ImportError:
            print("⚠️  Keras not installed, skipping LSTM. Using Isolation Forest only.")
            return None
        
        print(f"\n🧠 Training LSTM (lookback={lookback} cycles)...")
        
        X, y, feature_keys = self._prepare_feature_matrix()
        X_scaled = self.scaler.fit_transform(X)
        
        # Build sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - lookback):
            X_seq.append(X_scaled[i:i+lookback])
            # Target: is next cycle anomalous? (stage > 2)
            y_seq.append(1 if y[i+lookback] > 2 else 0)
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Build model
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(lookback, len(feature_keys))),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        # Train
        history = model.fit(X_seq, y_seq, epochs=50, batch_size=16, 
                           validation_split=0.2, verbose=0)
        
        print(f"   ✓ Training loss: {history.history['loss'][-1]:.4f}")
        print(f"   ✓ Validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
        
        return model
    
    def baseline_voltage_detection(self, voltage_threshold_pct_fade=20):
        """
        Classical BMS approach: detect degradation via capacity fade.
        Represents current industry baseline.
        
        Args:
            voltage_threshold_pct_fade: Trigger alarm at N% capacity fade
        """
        print(f"\n⚡ Classical Baseline: Voltage/Capacity Threshold")
        print(f"   Alarm threshold: {voltage_threshold_pct_fade}% capacity fade")
        
        baseline_scores = []
        baseline_detected_cycle = None
        
        for label in self.labels:
            # Score: 0 if below threshold, 1 if above
            fade = label['capacity_fade_pct']
            score = 1.0 if fade >= voltage_threshold_pct_fade else 0.0
            
            if score > 0.5 and baseline_detected_cycle is None:
                baseline_detected_cycle = label['cycle']
            
            baseline_scores.append(score)
        
        print(f"   Detected at cycle: {baseline_detected_cycle if baseline_detected_cycle else 'Never'}")
        
        return np.array(baseline_scores)
    
    def export_results(self, filename='anomaly_scores.json'):
        """Export anomaly scores for visualization."""
        results = {
            'anomaly_scores': self.anomaly_scores.tolist(),
            'cycles': [l['cycle'] for l in self.labels],
            'stages': [l['stage'] for l in self.labels],
            'capacity_fade_pct': [l['capacity_fade_pct'] for l in self.labels],
        }
        with open(filename, 'w') as f:
            json.dump(results, f)
        print(f"💾 Anomaly scores exported to {filename}")
    
    def plot_detection_comparison(self, save_fig='anomaly_detection_comparison.png'):
        """Compare model detection vs baseline."""
        cycles = [l['cycle'] for l in self.labels]
        capacity_fade = [l['capacity_fade_pct'] for l in self.labels]
        stages = [l['stage'] for l in self.labels]
        
        baseline_scores = self.baseline_voltage_detection()
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # Plot 1: Anomaly scores
        ax = axes[0]
        ax.plot(cycles, self.anomaly_scores, 'b-', linewidth=2, label='ML Anomaly Score')
        ax.axhline(0.5, color='red', linestyle='--', label='Detection Threshold')
        ax.fill_between(cycles, 0, 1, where=np.array(self.anomaly_scores) > 0.5, 
                        alpha=0.2, color='red', label='Anomaly Detected')
        ax.set_ylabel('Anomaly Score', fontsize=11)
        ax.set_title('ML-Based Anomaly Detection (Isolation Forest)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 2: Baseline voltage detection
        ax = axes[1]
        ax.plot(cycles, baseline_scores, 'g-', linewidth=2, label='Classical Voltage-Based Detection')
        ax.fill_between(cycles, 0, 1, where=baseline_scores > 0.5, alpha=0.2, color='green')
        ax.set_ylabel('Alert Status', fontsize=11)
        ax.set_title('Classical BMS Detection (80% Capacity Fade Threshold)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 3: Capacity fade and degradation stages
        ax = axes[2]
        ax2 = ax.twinx()
        
        ax.plot(cycles, capacity_fade, 'k-', linewidth=2, label='Capacity Fade %')
        ax.axhline(20, color='orange', linestyle='--', alpha=0.7, label='80% Capacity (BMS Alarm)')
        
        # Color-code by stage
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        for stage in range(1, 6):
            stage_cycles = [c for c, s in zip(cycles, stages) if s == stage]
            stage_fades = [f for f, s in zip(capacity_fade, stages) if s == stage]
            ax2.scatter(stage_cycles, stage_fades, color=colors[stage-1], s=20, 
                       label=f'Stage {stage}', alpha=0.6)
        
        ax.set_ylabel('Capacity Fade (%)', fontsize=11)
        ax2.set_ylabel('Degradation Stage', fontsize=11)
        ax.set_xlabel('Cycle Number', fontsize=11)
        ax.set_title('Capacity Fade and Degradation Stage Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f"📈 Detection comparison plot saved to {save_fig}")
        plt.close()


if __name__ == "__main__":
    # First, generate EIS features from eis_data
    print("Loading EIS features...")
    with open('eis_spectra.json', 'r') as f:
        eis_data = json.load(f)
    
    # Extract features and save
    eis_features = []
    for spectrum in eis_data:
        feat = {
            'cycle': spectrum['cycle'],
            'Rs': spectrum['Rs_Ohm'],
            'Rct': spectrum['Rct_Ohm'],
            'sigma_warburg': spectrum['sigma_warburg_Ohm_Hz_neg05'],
            'arc_diameter': spectrum['arc_diameter_Ohm'],
            'peak_freq': spectrum['peak_freq_Hz'],
            'Rs_normalized': spectrum['Rs_Ohm'] / 0.01,
            'Rct_normalized': spectrum['Rct_Ohm'] / 0.05,
        }
        eis_features.append(feat)
    
    with open('eis_features.json', 'w') as f:
        json.dump(eis_features, f)
    
    # Train anomaly detector
    engine = AnomalyDetectionEngine('eis_features.json', 'degradation_labels.json')
    iso_forest, scores = engine.train_isolation_forest(contamination=0.15)
    baseline = engine.baseline_voltage_detection()
    engine.export_results()
    engine.plot_detection_comparison()
    
    print("\n✅ Anomaly detection training complete!")
