#!/usr/bin/env python3
"""
Roundabout Traffic Metrics Calculator

Calculates advanced traffic engineering metrics for roundabout analysis:
- Entry capacity, circulating flow, exiting flow
- Critical gap and follow-up time
- Capacity index and reserve capacity
- Performance evaluation metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class RoundaboutGeometry:
    """Roundabout geometric parameters"""
    inscribed_circle_diameter: float = 30.0  # meters
    entry_width: float = 4.0  # meters
    circulating_roadway_width: float = 6.0  # meters
    entry_radius: float = 15.0  # meters
    entry_angle: float = 30.0  # degrees
    
    # Calibration constants for capacity formula
    # Entry Capacity = A × e^(-B × Circulating Flow)
    capacity_constant_A: float = 1500.0  # vehicles/hour (base capacity)
    capacity_constant_B: float = 0.00075  # flow interaction coefficient
    
    # Gap acceptance parameters
    critical_gap: float = 4.1  # seconds (time needed to enter safely)
    follow_up_time: float = 2.6  # seconds (time between successive entries)
    
    # Heavy vehicle factor
    heavy_vehicle_factor: float = 2.0  # PCU (Passenger Car Unit) equivalent


@dataclass
class ApproachMetrics:
    """Traffic metrics for a single approach"""
    approach_name: str
    
    # Flow measurements (vehicles/hour)
    entering_flow: float = 0.0  # Qe - vehicles entering from this approach
    circulating_flow: float = 0.0  # Qc - vehicles yielding to
    exiting_flow: float = 0.0  # Qu - vehicles leaving at this leg
    
    # Calculated metrics
    entry_capacity: float = 0.0  # C - maximum entry rate
    capacity_index: float = 0.0  # CR% - Qe/C ratio
    reserve_capacity: float = 0.0  # RC - C - Qe
    
    # Performance metrics
    average_delay: float = 0.0  # seconds per vehicle
    queue_length: float = 0.0  # vehicles
    level_of_service: str = "A"  # A-F rating
    
    # Vehicle composition
    total_vehicles: int = 0
    heavy_vehicles: int = 0
    heavy_vehicle_percentage: float = 0.0
    pcu_adjusted_flow: float = 0.0  # Flow adjusted for heavy vehicles
    
    # Time-based metrics
    average_dwell_time: float = 0.0  # seconds
    entry_gap_times: List[float] = None  # List of gaps between entries
    
    def __post_init__(self):
        if self.entry_gap_times is None:
            self.entry_gap_times = []


@dataclass
class RoundaboutPerformance:
    """Overall roundabout performance metrics"""
    total_entering_flow: float = 0.0  # Sum of all approaches
    total_circulating_flow: float = 0.0
    total_exiting_flow: float = 0.0
    
    # System performance
    overall_capacity_index: float = 0.0
    worst_approach_capacity_index: float = 0.0
    worst_approach_name: str = ""
    
    # Feasibility checks
    is_feasible: bool = True
    feasibility_threshold: float = 1800.0  # vehicles/hour per approach
    feasibility_notes: List[str] = None
    
    # Level of Service
    overall_los: str = "A"
    critical_approach_los: str = "A"
    
    def __post_init__(self):
        if self.feasibility_notes is None:
            self.feasibility_notes = []


class RoundaboutMetricsCalculator:
    """Calculate comprehensive roundabout traffic metrics"""
    
    # Level of Service (LOS) thresholds based on capacity index
    LOS_THRESHOLDS = {
        'A': 0.50,  # CR% < 50%
        'B': 0.65,  # CR% < 65%
        'C': 0.80,  # CR% < 80%
        'D': 0.90,  # CR% < 90%
        'E': 1.00,  # CR% < 100%
        'F': float('inf')  # CR% >= 100%
    }
    
    def __init__(self, geometry: RoundaboutGeometry = None):
        """
        Initialize calculator with roundabout geometry
        
        Args:
            geometry: RoundaboutGeometry instance with design parameters
        """
        self.geometry = geometry or RoundaboutGeometry()
        self.approaches: Dict[str, ApproachMetrics] = {}
        self.performance: Optional[RoundaboutPerformance] = None
    
    def calculate_entry_capacity(self, circulating_flow: float) -> float:
        """
        Calculate entry capacity using exponential decay model
        
        Entry Capacity = A × e^(-B × Circulating Flow)
        
        Args:
            circulating_flow: Volume of circulating traffic (vehicles/hour)
            
        Returns:
            Entry capacity in vehicles/hour
        """
        A = self.geometry.capacity_constant_A
        B = self.geometry.capacity_constant_B
        capacity = A * np.exp(-B * circulating_flow)
        return max(0, capacity)
    
    def calculate_pcu_adjusted_flow(self, vehicle_count: int, heavy_vehicle_count: int,
                                   duration_hours: float) -> Tuple[float, float]:
        """
        Convert vehicle counts to PCU-adjusted flow rate
        
        Args:
            vehicle_count: Total number of vehicles
            heavy_vehicle_count: Number of heavy vehicles (trucks, buses)
            duration_hours: Observation period in hours
            
        Returns:
            Tuple of (pcu_adjusted_flow, heavy_vehicle_percentage)
        """
        if vehicle_count == 0:
            return 0.0, 0.0
        
        # Calculate passenger car units
        light_vehicles = vehicle_count - heavy_vehicle_count
        pcu_total = light_vehicles + (heavy_vehicle_count * self.geometry.heavy_vehicle_factor)
        
        # Flow rate in vehicles/hour
        pcu_flow = pcu_total / duration_hours if duration_hours > 0 else 0
        hv_percentage = (heavy_vehicle_count / vehicle_count) * 100
        
        return pcu_flow, hv_percentage
    
    def calculate_capacity_metrics(self, entering_flow: float, entry_capacity: float) -> Tuple[float, float]:
        """
        Calculate capacity index and reserve capacity
        
        Args:
            entering_flow: Actual entering flow (vehicles/hour)
            entry_capacity: Maximum entry capacity (vehicles/hour)
            
        Returns:
            Tuple of (capacity_index, reserve_capacity)
        """
        if entry_capacity <= 0:
            return 100.0, 0.0
        
        capacity_index = (entering_flow / entry_capacity) * 100  # As percentage
        reserve_capacity = max(0, entry_capacity - entering_flow)
        
        return capacity_index, reserve_capacity
    
    def determine_level_of_service(self, capacity_index: float) -> str:
        """
        Determine Level of Service (LOS) based on capacity index
        
        Args:
            capacity_index: Capacity index as percentage (0-100+)
            
        Returns:
            LOS rating (A-F)
        """
        # Convert percentage to ratio
        cr_ratio = capacity_index / 100
        
        for los, threshold in self.LOS_THRESHOLDS.items():
            if cr_ratio < threshold:
                return los
        return 'F'
    
    def estimate_delay(self, capacity_index: float, circulating_flow: float) -> float:
        """
        Estimate average vehicle delay using empirical formula
        
        Args:
            capacity_index: Capacity index as percentage
            circulating_flow: Circulating flow (vehicles/hour)
            
        Returns:
            Average delay in seconds per vehicle
        """
        cr_ratio = capacity_index / 100
        
        # Simplified delay estimation (based on HCM methods)
        if cr_ratio < 0.85:
            # Undersaturated conditions
            delay = 5 + (10 * cr_ratio)
        else:
            # Near or over capacity - exponential growth
            delay = 5 + (10 * cr_ratio) + (50 * (cr_ratio - 0.85) ** 2)
        
        # Adjust for circulating flow impedance
        delay += (circulating_flow / 1000) * 2
        
        return max(0, delay)
    
    def estimate_queue_length(self, entering_flow: float, capacity_index: float) -> float:
        """
        Estimate average queue length
        
        Args:
            entering_flow: Entering flow (vehicles/hour)
            capacity_index: Capacity index as percentage
            
        Returns:
            Average queue length in vehicles
        """
        cr_ratio = capacity_index / 100
        
        if cr_ratio < 0.85:
            # Stable queue
            queue = (entering_flow / 3600) * cr_ratio * 10
        else:
            # Growing queue
            queue = (entering_flow / 3600) * ((cr_ratio ** 3) / (1.01 - cr_ratio)) * 50
        
        return max(0, queue)
    
    def add_approach_from_zone_data(self, approach_name: str, 
                                    entry_count: int, 
                                    circulating_count: int,
                                    exit_count: int,
                                    duration_seconds: float,
                                    entry_dwell_times: List[float] = None,
                                    circulating_dwell_times: List[float] = None,
                                    heavy_vehicle_count: int = 0) -> ApproachMetrics:
        """
        Create approach metrics from zone counting data
        
        Args:
            approach_name: Name of the approach (e.g., "north", "south")
            entry_count: Number of vehicles entering from this approach
            circulating_count: Number of vehicles in circulation
            exit_count: Number of vehicles exiting at this leg
            duration_seconds: Video duration in seconds
            entry_dwell_times: List of time spent in entry zone (seconds)
            circulating_dwell_times: List of time spent in circulating zone (seconds)
            heavy_vehicle_count: Number of heavy vehicles
            
        Returns:
            ApproachMetrics object with calculated values
        """
        duration_hours = duration_seconds / 3600
        
        # Calculate flow rates (vehicles/hour)
        entering_flow = entry_count / duration_hours if duration_hours > 0 else 0
        circulating_flow = circulating_count / duration_hours if duration_hours > 0 else 0
        exiting_flow = exit_count / duration_hours if duration_hours > 0 else 0
        
        # Adjust for heavy vehicles
        pcu_flow, hv_percentage = self.calculate_pcu_adjusted_flow(
            entry_count, heavy_vehicle_count, duration_hours
        )
        
        # Calculate entry capacity
        entry_capacity = self.calculate_entry_capacity(circulating_flow)
        
        # Calculate capacity metrics
        capacity_index, reserve_capacity = self.calculate_capacity_metrics(
            entering_flow, entry_capacity
        )
        
        # Determine LOS
        los = self.determine_level_of_service(capacity_index)
        
        # Estimate performance metrics
        avg_delay = self.estimate_delay(capacity_index, circulating_flow)
        queue_length = self.estimate_queue_length(entering_flow, capacity_index)
        
        # Calculate average dwell time
        avg_dwell = np.mean(entry_dwell_times) if entry_dwell_times else 0.0
        
        # Calculate entry gap times (time between successive vehicles)
        entry_gaps = []
        if entry_dwell_times and len(entry_dwell_times) > 1:
            # Estimate gaps from dwell pattern (simplified)
            entry_gaps = [self.geometry.follow_up_time] * len(entry_dwell_times)
        
        metrics = ApproachMetrics(
            approach_name=approach_name,
            entering_flow=entering_flow,
            circulating_flow=circulating_flow,
            exiting_flow=exiting_flow,
            entry_capacity=entry_capacity,
            capacity_index=capacity_index,
            reserve_capacity=reserve_capacity,
            average_delay=avg_delay,
            queue_length=queue_length,
            level_of_service=los,
            total_vehicles=entry_count,
            heavy_vehicles=heavy_vehicle_count,
            heavy_vehicle_percentage=hv_percentage,
            pcu_adjusted_flow=pcu_flow,
            average_dwell_time=avg_dwell,
            entry_gap_times=entry_gaps
        )
        
        self.approaches[approach_name] = metrics
        return metrics
    
    def calculate_overall_performance(self) -> RoundaboutPerformance:
        """
        Calculate overall roundabout performance metrics
        
        Returns:
            RoundaboutPerformance object
        """
        if not self.approaches:
            return RoundaboutPerformance()
        
        # Sum flows across all approaches
        total_entering = sum(m.entering_flow for m in self.approaches.values())
        total_circulating = sum(m.circulating_flow for m in self.approaches.values())
        total_exiting = sum(m.exiting_flow for m in self.approaches.values())
        
        # Find worst approach
        worst_approach = max(self.approaches.values(), 
                           key=lambda m: m.capacity_index)
        
        # Calculate overall capacity index (weighted by flow)
        if total_entering > 0:
            overall_ci = sum(m.capacity_index * m.entering_flow 
                           for m in self.approaches.values()) / total_entering
        else:
            overall_ci = 0.0
        
        # Determine overall LOS
        overall_los = self.determine_level_of_service(overall_ci)
        
        # Feasibility checks
        feasibility_notes = []
        is_feasible = True
        threshold = self.geometry.capacity_constant_A * 1.2  # 120% of base capacity
        
        for name, metrics in self.approaches.items():
            combined_flow = metrics.entering_flow + metrics.circulating_flow
            if combined_flow > threshold:
                is_feasible = False
                feasibility_notes.append(
                    f"{name}: Combined flow ({combined_flow:.0f} veh/h) exceeds threshold ({threshold:.0f} veh/h)"
                )
            
            if metrics.capacity_index > 90:
                feasibility_notes.append(
                    f"{name}: High capacity utilization ({metrics.capacity_index:.1f}%) - approaching saturation"
                )
        
        if not feasibility_notes:
            feasibility_notes.append("All approaches within acceptable capacity limits")
        
        performance = RoundaboutPerformance(
            total_entering_flow=total_entering,
            total_circulating_flow=total_circulating,
            total_exiting_flow=total_exiting,
            overall_capacity_index=overall_ci,
            worst_approach_capacity_index=worst_approach.capacity_index,
            worst_approach_name=worst_approach.approach_name,
            is_feasible=is_feasible,
            feasibility_threshold=threshold,
            feasibility_notes=feasibility_notes,
            overall_los=overall_los,
            critical_approach_los=worst_approach.level_of_service
        )
        
        self.performance = performance
        return performance
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive metrics report
        
        Args:
            output_path: Optional path to save JSON report
            
        Returns:
            Dictionary with all metrics
        """
        # Ensure performance is calculated
        if self.performance is None:
            self.calculate_overall_performance()
        
        report = {
            'geometry': asdict(self.geometry),
            'approaches': {
                name: asdict(metrics) 
                for name, metrics in self.approaches.items()
            },
            'overall_performance': asdict(self.performance),
            'summary': {
                'total_approaches': len(self.approaches),
                'critical_approach': self.performance.worst_approach_name,
                'overall_los': self.performance.overall_los,
                'is_feasible': self.performance.is_feasible,
                'total_entering_vehicles': sum(m.total_vehicles for m in self.approaches.values()),
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Metrics report saved to: {output_path}")
        
        return report
    
    def print_summary(self):
        """Print formatted summary of metrics"""
        if not self.performance:
            self.calculate_overall_performance()
        
        print("\n" + "=" * 80)
        print("ROUNDABOUT TRAFFIC ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\nGeometry Parameters:")
        print(f"  Inscribed Circle Diameter: {self.geometry.inscribed_circle_diameter} m")
        print(f"  Critical Gap: {self.geometry.critical_gap} s")
        print(f"  Follow-up Time: {self.geometry.follow_up_time} s")
        print(f"  Capacity Constants: A={self.geometry.capacity_constant_A}, B={self.geometry.capacity_constant_B}")
        
        print(f"\n{'Approach Metrics':-^80}")
        for name, metrics in sorted(self.approaches.items()):
            print(f"\n{name.upper()}:")
            print(f"  Entering Flow (Qe):      {metrics.entering_flow:7.1f} veh/h")
            print(f"  Circulating Flow (Qc):   {metrics.circulating_flow:7.1f} veh/h")
            print(f"  Exiting Flow (Qu):       {metrics.exiting_flow:7.1f} veh/h")
            print(f"  Entry Capacity (C):      {metrics.entry_capacity:7.1f} veh/h")
            print(f"  Capacity Index (CR%):    {metrics.capacity_index:7.1f} %")
            print(f"  Reserve Capacity (RC):   {metrics.reserve_capacity:7.1f} veh/h")
            print(f"  Level of Service:        {metrics.level_of_service}")
            print(f"  Average Delay:           {metrics.average_delay:7.1f} seconds")
            print(f"  Queue Length:            {metrics.queue_length:7.1f} vehicles")
            if metrics.heavy_vehicle_percentage > 0:
                print(f"  Heavy Vehicles:          {metrics.heavy_vehicle_percentage:7.1f} %")
        
        print(f"\n{'Overall Performance':-^80}")
        print(f"  Total Entering Flow:     {self.performance.total_entering_flow:7.1f} veh/h")
        print(f"  Total Circulating Flow:  {self.performance.total_circulating_flow:7.1f} veh/h")
        print(f"  Total Exiting Flow:      {self.performance.total_exiting_flow:7.1f} veh/h")
        print(f"  Overall Capacity Index:  {self.performance.overall_capacity_index:7.1f} %")
        print(f"  Overall Level of Service: {self.performance.overall_los}")
        print(f"  Critical Approach:       {self.performance.worst_approach_name} (LOS {self.performance.critical_approach_los})")
        print(f"  Feasibility:             {'✓ FEASIBLE' if self.performance.is_feasible else '✗ NOT FEASIBLE'}")
        
        print(f"\n{'Feasibility Assessment':-^80}")
        for note in self.performance.feasibility_notes:
            print(f"  • {note}")
        
        print("\n" + "=" * 80)


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate roundabout traffic metrics")
    parser.add_argument("counts_file", help="Path to counts JSON file from vehicle_counting.py")
    parser.add_argument("--output", help="Path to save metrics report JSON")
    parser.add_argument("--print-summary", action="store_true", help="Print formatted summary")
    
    args = parser.parse_args()
    
    # Load counts data
    with open(args.counts_file, 'r') as f:
        data = json.load(f)
    
    # Initialize calculator
    calculator = RoundaboutMetricsCalculator()
    
    # Extract zone data and calculate metrics for each approach
    zones = data.get('zones', {})
    dwell_times = data.get('dwell_times', {})
    duration = data.get('total_frames', 0) / data.get('fps', 1)
    
    # Group zones by approach (assumes naming like north_entry, north_circulating, etc.)
    approaches = set()
    for zone_name in zones.keys():
        approach = zone_name.split('_')[0]
        approaches.add(approach)
    
    for approach in approaches:
        entry_count = zones.get(f'{approach}_entry', 0)
        circ_count = zones.get(f'{approach}_circulating', 0)
        exit_count = zones.get(f'{approach}_exit', 0)
        
        entry_dwell = dwell_times.get(f'{approach}_entry', {})
        circ_dwell = dwell_times.get(f'{approach}_circulating', {})
        
        # Get dwell time samples if available
        entry_times = []
        if 'mean' in entry_dwell and 'count' in entry_dwell:
            # Approximate samples from statistics
            entry_times = [entry_dwell['mean']] * entry_dwell['count']
        
        calculator.add_approach_from_zone_data(
            approach_name=approach,
            entry_count=entry_count,
            circulating_count=circ_count,
            exit_count=exit_count,
            duration_seconds=duration,
            entry_dwell_times=entry_times
        )
    
    # Calculate overall performance
    calculator.calculate_overall_performance()
    
    # Generate report
    report = calculator.generate_report(args.output)
    
    # Print summary if requested
    if args.print_summary:
        calculator.print_summary()


if __name__ == "__main__":
    main()
