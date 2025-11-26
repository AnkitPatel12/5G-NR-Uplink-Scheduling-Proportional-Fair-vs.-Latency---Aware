"""
5G NR Uplink Scheduler Simulation
Comparing Proportional Fair vs. Latency-Aware Scheduling
for Mixed eMBB/URLLC Traffic
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

@dataclass
class SimConfig:
    """Simulation configuration parameters"""
    # Time parameters
    slot_duration_ms: float = 1.0  # Each scheduling slot is 1ms
    simulation_duration_ms: int = 10000  # 10 seconds of simulation
    
    # Network parameters
    num_resource_blocks: int = 50  # Available frequency resources
    num_embb_users: int = 10  # Number of eMBB users
    num_urllc_users: int = 5  # Number of URLLC users
    
    # Traffic parameters (eMBB)
    embb_packet_size_bytes: int = 1500  # Average packet size
    embb_arrival_rate: float = 0.3  # Packets per ms (Poisson)
    embb_burstiness: float = 1.0  # 1.0 = Poisson, >1 = bursty
    
    # Traffic parameters (URLLC)
    urllc_packet_size_bytes: int = 100  # Smaller packets
    urllc_arrival_rate: float = 0.1  # Packets per ms
    urllc_deadline_ms: float = 5.0  # Strict 5ms deadline
    urllc_burstiness: float = 1.0
    
    # Channel model parameters
    channel_quality_mean_db: float = 10.0  # Average SNR in dB
    channel_quality_std_db: float = 5.0  # Channel variation
    fading_correlation: float = 0.9  # Temporal correlation


# ============================================================================
# USER CLASS
# ============================================================================

class User:
    """Represents a single user (eMBB or URLLC)"""
    
    def __init__(self, user_id: int, user_type: str, config: SimConfig):
        self.user_id = user_id
        self.user_type = user_type  # 'eMBB' or 'URLLC'
        self.config = config
        
        # Queue of packets waiting to be transmitted
        self.packet_queue = deque()
        
        # Channel quality (SNR in dB)
        self.channel_quality_db = config.channel_quality_mean_db
        
        # Statistics
        self.total_packets_generated = 0
        self.total_packets_transmitted = 0
        self.total_bytes_transmitted = 0
        self.total_delay_sum = 0.0
        self.missed_deadlines = 0
        self.average_throughput_history = []
        
    def generate_traffic(self, current_time_ms: float):
        """Generate new packets for this user"""
        if self.user_type == 'eMBB':
            arrival_rate = self.config.embb_arrival_rate
            packet_size = self.config.embb_packet_size_bytes
            burstiness = self.config.embb_burstiness
            deadline = None  # No strict deadline
        else:  # URLLC
            arrival_rate = self.config.urllc_arrival_rate
            packet_size = self.config.urllc_packet_size_bytes
            burstiness = self.config.urllc_burstiness
            deadline = current_time_ms + self.config.urllc_deadline_ms
        
        # Generate packets using Pareto distribution for burstiness
        # When burstiness = 1, approaches Poisson
        shape = 1.0 / burstiness
        num_packets = np.random.gamma(shape, arrival_rate / shape)
        num_packets = int(np.random.poisson(num_packets))
        
        for _ in range(num_packets):
            packet = {
                'arrival_time': current_time_ms,
                'size_bytes': packet_size,
                'deadline': deadline,
                'user_type': self.user_type
            }
            self.packet_queue.append(packet)
            self.total_packets_generated += 1
    
    def update_channel_quality(self):
        """Update channel quality with temporal correlation (fading)"""
        # Correlated Gaussian process
        innovation = np.random.normal(0, self.config.channel_quality_std_db)
        self.channel_quality_db = (
            self.config.fading_correlation * self.channel_quality_db +
            (1 - self.config.fading_correlation) * self.config.channel_quality_mean_db +
            np.sqrt(1 - self.config.fading_correlation**2) * innovation
        )
    
    def get_data_rate_mbps(self, num_rbs: int) -> float:
        """Calculate achievable data rate based on channel quality and RBs"""
        # Shannon capacity: R = B * log2(1 + SNR)
        # Each RB is ~180 kHz bandwidth
        bandwidth_mhz = num_rbs * 0.18
        snr_linear = 10 ** (self.channel_quality_db / 10)
        rate_mbps = bandwidth_mhz * np.log2(1 + snr_linear)
        return rate_mbps
    
    def get_head_of_line_delay(self, current_time_ms: float) -> float:
        """Get delay of the oldest packet in queue"""
        if not self.packet_queue:
            return 0.0
        return current_time_ms - self.packet_queue[0]['arrival_time']
    
    def check_and_remove_expired_packets(self, current_time_ms: float):
        """Remove URLLC packets that missed their deadline"""
        if self.user_type != 'URLLC':
            return
        
        while self.packet_queue:
            packet = self.packet_queue[0]
            if packet['deadline'] and current_time_ms > packet['deadline']:
                self.packet_queue.popleft()
                self.missed_deadlines += 1
            else:
                break


# ============================================================================
# SCHEDULER BASE CLASS
# ============================================================================

class Scheduler:
    """Base class for scheduling algorithms"""
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.name = "BaseScheduler"
    
    def schedule(self, users: List[User], current_time_ms: float) -> dict:
        """
        Decide resource allocation for this slot
        Returns: {user_id: num_resource_blocks}
        """
        raise NotImplementedError


# ============================================================================
# PROPORTIONAL FAIR SCHEDULER
# ============================================================================

class ProportionalFairScheduler(Scheduler):
    """Proportional Fair Scheduler - balances throughput and fairness"""
    
    def __init__(self, config: SimConfig):
        super().__init__(config)
        self.name = "Proportional Fair"
        # Track average throughput for each user (exponentially weighted)
        self.avg_throughput = {}
        self.alpha = 0.95  # Smoothing factor
    
    def schedule(self, users: List[User], current_time_ms: float) -> dict:
        """
        Proportional Fair: maximize sum of log(throughput)
        Metric: instantaneous_rate / average_throughput
        """
        allocation = {}
        
        # Calculate metrics for all users with data to send
        user_metrics = []
        for user in users:
            if not user.packet_queue:
                continue
            
            # Initialize average throughput if needed
            if user.user_id not in self.avg_throughput:
                self.avg_throughput[user.user_id] = 0.001  # Small initial value
            
            # Instantaneous rate if given all RBs
            inst_rate = user.get_data_rate_mbps(self.config.num_resource_blocks)
            avg_rate = self.avg_throughput[user.user_id]
            
            # PF metric
            pf_metric = inst_rate / (avg_rate + 0.001)
            
            user_metrics.append({
                'user': user,
                'metric': pf_metric,
                'inst_rate': inst_rate
            })
        
        # Sort by metric (descending)
        user_metrics.sort(key=lambda x: x['metric'], reverse=True)
        
        # Allocate resources to top users
        remaining_rbs = self.config.num_resource_blocks
        for item in user_metrics:
            if remaining_rbs <= 0:
                break
            
            user = item['user']
            # Allocate equal share or remaining RBs
            allocated_rbs = min(remaining_rbs, 
                               self.config.num_resource_blocks // max(len(user_metrics), 1))
            allocation[user.user_id] = allocated_rbs
            remaining_rbs -= allocated_rbs
            
            # Update average throughput
            actual_rate = user.get_data_rate_mbps(allocated_rbs)
            self.avg_throughput[user.user_id] = (
                self.alpha * self.avg_throughput[user.user_id] +
                (1 - self.alpha) * actual_rate
            )
        
        return allocation


# ============================================================================
# LATENCY-AWARE SCHEDULER
# ============================================================================

class LatencyAwareScheduler(Scheduler):
    """Latency-Aware Scheduler - prioritizes packets close to deadline"""
    
    def __init__(self, config: SimConfig):
        super().__init__(config)
        self.name = "Latency-Aware"
    
    def schedule(self, users: List[User], current_time_ms: float) -> dict:
        """
        Latency-Aware: prioritize based on urgency
        Metric: weighted delay (URLLC gets higher weight)
        """
        allocation = {}
        
        # Calculate urgency for all users with data
        user_metrics = []
        for user in users:
            if not user.packet_queue:
                continue
            
            hol_delay = user.get_head_of_line_delay(current_time_ms)
            
            # Calculate urgency metric
            if user.user_type == 'URLLC':
                packet = user.packet_queue[0]
                time_to_deadline = packet['deadline'] - current_time_ms
                # Higher urgency as deadline approaches
                urgency = 1000.0 / (time_to_deadline + 0.1)
            else:  # eMBB
                # Lower priority, but still consider delay
                urgency = hol_delay / 10.0
            
            user_metrics.append({
                'user': user,
                'urgency': urgency,
                'hol_delay': hol_delay
            })
        
        # Sort by urgency (descending)
        user_metrics.sort(key=lambda x: x['urgency'], reverse=True)
        
        # Allocate resources prioritizing urgent users
        remaining_rbs = self.config.num_resource_blocks
        
        # First pass: ensure URLLC users get resources
        urllc_users = [item for item in user_metrics if item['user'].user_type == 'URLLC']
        for item in urllc_users:
            if remaining_rbs <= 0:
                break
            user = item['user']
            # Give URLLC users priority allocation
            allocated_rbs = min(remaining_rbs, 
                               max(5, self.config.num_resource_blocks // 4))
            allocation[user.user_id] = allocated_rbs
            remaining_rbs -= allocated_rbs
        
        # Second pass: allocate remaining to others
        embb_users = [item for item in user_metrics if item['user'].user_type == 'eMBB']
        for item in embb_users:
            if remaining_rbs <= 0:
                break
            user = item['user']
            allocated_rbs = min(remaining_rbs,
                               self.config.num_resource_blocks // max(len(embb_users), 1))
            allocation[user.user_id] = allocated_rbs
            remaining_rbs -= allocated_rbs
        
        return allocation


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class Simulator:
    """Main simulation engine"""
    
    def __init__(self, config: SimConfig, scheduler: Scheduler):
        self.config = config
        self.scheduler = scheduler
        
        # Create users
        self.users = []
        for i in range(config.num_embb_users):
            self.users.append(User(i, 'eMBB', config))
        for i in range(config.num_urllc_users):
            self.users.append(User(config.num_embb_users + i, 'URLLC', config))
        
        # Statistics
        self.time_slots = []
        self.total_throughput_history = []
        self.urllc_delay_history = []
        self.embb_delay_history = []
    
    def run(self):
        """Run the simulation"""
        num_slots = int(self.config.simulation_duration_ms / self.config.slot_duration_ms)
        
        print(f"\nRunning simulation with {self.scheduler.name} scheduler...")
        print(f"Duration: {self.config.simulation_duration_ms}ms, Slots: {num_slots}")
        
        for slot in range(num_slots):
            current_time = slot * self.config.slot_duration_ms
            
            # Progress indicator
            if slot % 1000 == 0:
                print(f"  Progress: {slot}/{num_slots} slots ({100*slot/num_slots:.1f}%)")
            
            # Step 1: Generate new traffic for all users
            for user in self.users:
                user.generate_traffic(current_time)
            
            # Step 2: Update channel quality
            for user in self.users:
                user.update_channel_quality()
            
            # Step 3: Remove expired URLLC packets
            for user in self.users:
                user.check_and_remove_expired_packets(current_time)
            
            # Step 4: Run scheduler
            allocation = self.scheduler.schedule(self.users, current_time)
            
            # Step 5: Transmit data according to allocation
            slot_throughput = 0.0
            for user_id, num_rbs in allocation.items():
                user = self.users[user_id]
                if not user.packet_queue or num_rbs <= 0:
                    continue
                
                # Calculate how much can be transmitted
                rate_mbps = user.get_data_rate_mbps(num_rbs)
                bytes_can_transmit = rate_mbps * 1e6 / 8 * self.config.slot_duration_ms / 1000
                
                # Transmit packets
                bytes_transmitted = 0
                packets_transmitted = 0
                while user.packet_queue and bytes_transmitted < bytes_can_transmit:
                    packet = user.packet_queue.popleft()
                    bytes_transmitted += packet['size_bytes']
                    packets_transmitted += 1
                    
                    # Record delay
                    delay = current_time - packet['arrival_time']
                    user.total_delay_sum += delay
                    user.total_packets_transmitted += 1
                
                user.total_bytes_transmitted += bytes_transmitted
                slot_throughput += bytes_transmitted * 8 / 1e6  # Convert to Mbps
            
            # Record statistics
            self.time_slots.append(current_time)
            self.total_throughput_history.append(slot_throughput)
            
            # Record delays
            urllc_delays = [u.get_head_of_line_delay(current_time) 
                           for u in self.users if u.user_type == 'URLLC' and u.packet_queue]
            embb_delays = [u.get_head_of_line_delay(current_time) 
                          for u in self.users if u.user_type == 'eMBB' and u.packet_queue]
            
            self.urllc_delay_history.append(np.mean(urllc_delays) if urllc_delays else 0)
            self.embb_delay_history.append(np.mean(embb_delays) if embb_delays else 0)
        
        print(f"  Simulation complete!")
    
    def get_results(self) -> dict:
        """Compile simulation results"""
        results = {
            'scheduler_name': self.scheduler.name,
            'config': self.config,
        }
        
        # Aggregate user statistics
        embb_users = [u for u in self.users if u.user_type == 'eMBB']
        urllc_users = [u for u in self.users if u.user_type == 'URLLC']
        
        # eMBB metrics
        embb_throughput = sum(u.total_bytes_transmitted for u in embb_users) * 8 / \
                         (self.config.simulation_duration_ms / 1000) / 1e6
        embb_avg_delay = np.mean([u.total_delay_sum / max(u.total_packets_transmitted, 1) 
                                  for u in embb_users])
        
        # URLLC metrics
        urllc_throughput = sum(u.total_bytes_transmitted for u in urllc_users) * 8 / \
                          (self.config.simulation_duration_ms / 1000) / 1e6
        urllc_avg_delay = np.mean([u.total_delay_sum / max(u.total_packets_transmitted, 1) 
                                   for u in urllc_users])
        urllc_missed = sum(u.missed_deadlines for u in urllc_users)
        urllc_total = sum(u.total_packets_generated for u in urllc_users)
        urllc_miss_rate = urllc_missed / max(urllc_total, 1) * 100
        
        results.update({
            'embb_throughput_mbps': embb_throughput,
            'embb_avg_delay_ms': embb_avg_delay,
            'urllc_throughput_mbps': urllc_throughput,
            'urllc_avg_delay_ms': urllc_avg_delay,
            'urllc_missed_deadlines': urllc_missed,
            'urllc_miss_rate_percent': urllc_miss_rate,
            'total_throughput_mbps': embb_throughput + urllc_throughput,
            'time_slots': self.time_slots,
            'throughput_history': self.total_throughput_history,
            'urllc_delay_history': self.urllc_delay_history,
            'embb_delay_history': self.embb_delay_history,
        })
        
        return results


# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def plot_results(results_list: List[dict], scenario_name: str):
    """Create comprehensive plots comparing schedulers"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'5G Uplink Scheduler Comparison - {scenario_name}', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Plot 1: Throughput over time
    ax = axes[0, 0]
    for i, result in enumerate(results_list):
        # Smooth the throughput
        window = 100
        smoothed = np.convolve(result['throughput_history'], 
                              np.ones(window)/window, mode='valid')
        time_smooth = result['time_slots'][:len(smoothed)]
        ax.plot(time_smooth, smoothed, label=result['scheduler_name'], 
               color=colors[i], linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.set_title('Instantaneous Throughput', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: URLLC Delay over time
    ax = axes[0, 1]
    for i, result in enumerate(results_list):
        window = 100
        smoothed = np.convolve(result['urllc_delay_history'], 
                              np.ones(window)/window, mode='valid')
        time_smooth = result['time_slots'][:len(smoothed)]
        ax.plot(time_smooth, smoothed, label=result['scheduler_name'],
               color=colors[i], linewidth=2, alpha=0.8)
    ax.axhline(y=5, color='red', linestyle='--', label='Deadline', linewidth=2)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Head-of-Line Delay (ms)', fontsize=11)
    ax.set_title('URLLC Packet Delay', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: eMBB Delay over time
    ax = axes[0, 2]
    for i, result in enumerate(results_list):
        window = 100
        smoothed = np.convolve(result['embb_delay_history'], 
                              np.ones(window)/window, mode='valid')
        time_smooth = result['time_slots'][:len(smoothed)]
        ax.plot(time_smooth, smoothed, label=result['scheduler_name'],
               color=colors[i], linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Head-of-Line Delay (ms)', fontsize=11)
    ax.set_title('eMBB Packet Delay', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Bar chart - Total Throughput
    ax = axes[1, 0]
    schedulers = [r['scheduler_name'] for r in results_list]
    throughputs = [r['total_throughput_mbps'] for r in results_list]
    bars = ax.bar(schedulers, throughputs, color=colors[:len(schedulers)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.set_title('Total System Throughput', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 5: Bar chart - URLLC Miss Rate
    ax = axes[1, 1]
    miss_rates = [r['urllc_miss_rate_percent'] for r in results_list]
    bars = ax.bar(schedulers, miss_rates, color=colors[:len(schedulers)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Deadline Miss Rate (%)', fontsize=11)
    ax.set_title('URLLC Deadline Miss Rate', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 6: Bar chart - Average Delays
    ax = axes[1, 2]
    x = np.arange(len(schedulers))
    width = 0.35
    urllc_delays = [r['urllc_avg_delay_ms'] for r in results_list]
    embb_delays = [r['embb_avg_delay_ms'] for r in results_list]
    
    bars1 = ax.bar(x - width/2, urllc_delays, width, label='URLLC', 
                   color='#E63946', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, embb_delays, width, label='eMBB',
                   color='#457B9D', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Average Delay (ms)', fontsize=11)
    ax.set_title('Average Packet Delay', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_results_table(results_list: List[dict]):
    """Print formatted results table"""
    print("\n" + "="*80)
    print(" SIMULATION RESULTS SUMMARY")
    print("="*80)
    
    for result in results_list:
        print(f"\n{result['scheduler_name']} Scheduler:")
        print("-" * 80)
        print(f"  Total System Throughput:     {result['total_throughput_mbps']:.2f} Mbps")
        print(f"  eMBB Throughput:             {result['embb_throughput_mbps']:.2f} Mbps")
        print(f"  eMBB Average Delay:          {result['embb_avg_delay_ms']:.2f} ms")
        print(f"  URLLC Throughput:            {result['urllc_throughput_mbps']:.2f} Mbps")
        print(f"  URLLC Average Delay:         {result['urllc_avg_delay_ms']:.2f} ms")
        print(f"  URLLC Missed Deadlines:      {result['urllc_missed_deadlines']}")
        print(f"  URLLC Deadline Miss Rate:    {result['urllc_miss_rate_percent']:.2f}%")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_scenario(config: SimConfig, scenario_name: str) -> List[dict]:
    """Run both schedulers for a given scenario"""
    print(f"\n{'='*80}")
    print(f" SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    
    results = []
    
    # Run Proportional Fair
    pf_scheduler = ProportionalFairScheduler(config)
    sim_pf = Simulator(config, pf_scheduler)
    sim_pf.run()
    results.append(sim_pf.get_results())
    
    # Run Latency-Aware
    la_scheduler = LatencyAwareScheduler(config)
    sim_la = Simulator(config, la_scheduler)
    sim_la.run()
    results.append(sim_la.get_results())
    
    return results


def main():
    """Main function to run all scenarios"""
    
    print("\n" + "="*80)
    print(" 5G NR UPLINK SCHEDULING SIMULATION")
    print(" Comparing Proportional Fair vs. Latency-Aware Schedulers")
    print("="*80)
    
    # ========================================================================
    # SCENARIO 1: Baseline (moderate traffic, good channel)
    # ========================================================================
    config1 = SimConfig(
        simulation_duration_ms=10000,
        num_embb_users=10,
        num_urllc_users=5,
        embb_burstiness=1.0,
        urllc_burstiness=1.0,
        channel_quality_mean_db=10.0,
        channel_quality_std_db=3.0
    )
    results1 = run_scenario(config1, "Baseline: Moderate Traffic, Good Channel")
    print_results_table(results1)
    fig1 = plot_results(results1, "Baseline Scenario")
    plt.savefig('/mnt/user-data/outputs/scenario1_baseline.png', dpi=300, bbox_inches='tight')
    print("Saved: scenario1_baseline.png")
    
    # ========================================================================
    # SCENARIO 2: High Burstiness
    # ========================================================================
    config2 = SimConfig(
        simulation_duration_ms=10000,
        num_embb_users=10,
        num_urllc_users=5,
        embb_burstiness=3.0,  # Very bursty!
        urllc_burstiness=2.0,
        channel_quality_mean_db=10.0,
        channel_quality_std_db=3.0
    )
    results2 = run_scenario(config2, "High Burstiness: Challenging Traffic")
    print_results_table(results2)
    fig2 = plot_results(results2, "High Burstiness")
    plt.savefig('/mnt/user-data/outputs/scenario2_burstiness.png', dpi=300, bbox_inches='tight')
    print("Saved: scenario2_burstiness.png")
    
    # ========================================================================
    # SCENARIO 3: Poor Channel Quality
    # ========================================================================
    config3 = SimConfig(
        simulation_duration_ms=10000,
        num_embb_users=10,
        num_urllc_users=5,
        embb_burstiness=1.0,
        urllc_burstiness=1.0,
        channel_quality_mean_db=5.0,  # Lower SNR
        channel_quality_std_db=5.0    # High variation
    )
    results3 = run_scenario(config3, "Poor Channel Quality: Low SNR & High Fading")
    print_results_table(results3)
    fig3 = plot_results(results3, "Poor Channel Quality")
    plt.savefig('/mnt/user-data/outputs/scenario3_channel.png', dpi=300, bbox_inches='tight')
    print("Saved: scenario3_channel.png")
    
    # ========================================================================
    # SCENARIO 4: Heavy Load
    # ========================================================================
    config4 = SimConfig(
        simulation_duration_ms=10000,
        num_embb_users=15,  # More users!
        num_urllc_users=8,
        embb_arrival_rate=0.5,  # Higher traffic
        urllc_arrival_rate=0.2,
        embb_burstiness=1.5,
        urllc_burstiness=1.5,
        channel_quality_mean_db=10.0,
        channel_quality_std_db=3.0
    )
    results4 = run_scenario(config4, "Heavy Load: More Users & Higher Traffic")
    print_results_table(results4)
    fig4 = plot_results(results4, "Heavy Load")
    plt.savefig('/mnt/user-data/outputs/scenario4_heavy_load.png', dpi=300, bbox_inches='tight')
    print("Saved: scenario4_heavy_load.png")
    
    # ========================================================================
    # Create Comparison Summary
    # ========================================================================
    create_comparison_summary([results1, results2, results3, results4],
                             ['Baseline', 'High Burstiness', 'Poor Channel', 'Heavy Load'])
    
    print("\n" + "="*80)
    print(" SIMULATION COMPLETE!")
    print(" All results saved to /mnt/user-data/outputs/")
    print("="*80 + "\n")


def create_comparison_summary(all_results: List[List[dict]], scenario_names: List[str]):
    """Create a summary comparison across all scenarios"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Scenario Comparison: PF vs. Latency-Aware', 
                 fontsize=16, fontweight='bold')
    
    # Extract data
    scenarios = scenario_names
    pf_throughput = [r[0]['total_throughput_mbps'] for r in all_results]
    la_throughput = [r[1]['total_throughput_mbps'] for r in all_results]
    pf_urllc_delay = [r[0]['urllc_avg_delay_ms'] for r in all_results]
    la_urllc_delay = [r[1]['urllc_avg_delay_ms'] for r in all_results]
    pf_miss_rate = [r[0]['urllc_miss_rate_percent'] for r in all_results]
    la_miss_rate = [r[1]['urllc_miss_rate_percent'] for r in all_results]
    pf_embb_delay = [r[0]['embb_avg_delay_ms'] for r in all_results]
    la_embb_delay = [r[1]['embb_avg_delay_ms'] for r in all_results]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Plot 1: Throughput
    ax = axes[0, 0]
    bars1 = ax.bar(x - width/2, pf_throughput, width, label='Proportional Fair',
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, la_throughput, width, label='Latency-Aware',
                   color='#A23B72', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Total Throughput (Mbps)', fontsize=12)
    ax.set_title('System Throughput Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: URLLC Delay
    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, pf_urllc_delay, width, label='Proportional Fair',
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, la_urllc_delay, width, label='Latency-Aware',
                   color='#A23B72', alpha=0.8, edgecolor='black')
    ax.axhline(y=5, color='red', linestyle='--', label='Target', linewidth=2)
    ax.set_ylabel('Average URLLC Delay (ms)', fontsize=12)
    ax.set_title('URLLC Latency Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Miss Rate
    ax = axes[1, 0]
    bars1 = ax.bar(x - width/2, pf_miss_rate, width, label='Proportional Fair',
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, la_miss_rate, width, label='Latency-Aware',
                   color='#A23B72', alpha=0.8, edgecolor='black')
    ax.set_ylabel('URLLC Deadline Miss Rate (%)', fontsize=12)
    ax.set_title('Reliability Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 4: eMBB Delay
    ax = axes[1, 1]
    bars1 = ax.bar(x - width/2, pf_embb_delay, width, label='Proportional Fair',
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, la_embb_delay, width, label='Latency-Aware',
                   color='#A23B72', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average eMBB Delay (ms)', fontsize=12)
    ax.set_title('eMBB Latency Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/comparison_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: comparison_summary.png")
    plt.close()


if __name__ == "__main__":
    main()
