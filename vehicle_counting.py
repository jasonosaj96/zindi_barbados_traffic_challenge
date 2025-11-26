#!/usr/bin/env python3
"""
Vehicle Counting Script for Roundabout Traffic Analysis

This script processes video footage of a roundabout from different camera angles,
detects vehicles using YOLO, and counts them in specific zones defined by polygons.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict

# Import roundabout metrics calculator
try:
    from roundabout_metrics import RoundaboutMetricsCalculator, RoundaboutGeometry
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: roundabout_metrics.py not found. Advanced metrics will not be calculated.")


class VehicleCounter:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        camera_config: Dict = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        enable_tracking: bool = True
    ):
        self.model = YOLO(model_path)
        self.camera_config = camera_config or {}
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.enable_tracking = enable_tracking
        # add type hints
        self.zones: Dict[str, sv.PolygonZone] = {}
        self.zone_annotators: Dict[str, sv.PolygonZoneAnnotator] = {}
        self.counts: defaultdict[str, int] = defaultdict(int)

        # Tracking components
        self.tracker = sv.ByteTrack() if enable_tracking else None
        # Track vehicle states: {tracker_id: {zone_name: {'enter_time': frame_num, 'enter_timestamp': seconds}}}
        self.vehicle_states: Dict[int, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
        # Store completed dwell times: {zone_name: [dwell_times_in_seconds]}
        self.dwell_times: Dict[str, List[float]] = defaultdict(list)

        if self.camera_config:
            self._setup_zones()

    def _setup_zones(self):
        """Initialize polygon zones from camera configuration."""
        for zone_name, polygon_points in self.camera_config.get("zones", {}).items():
            polygon = np.array(polygon_points, dtype=np.int32)
            self.zones[zone_name] = sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=[sv.Position.BOTTOM_CENTER]
            )
            self.zone_annotators[zone_name] = sv.PolygonZoneAnnotator(
                zone=self.zones[zone_name],
                color=sv.Color.from_hex(self.camera_config.get("zone_colors", {}).get(zone_name, "#FF0000")),
                thickness=2,
                text_thickness=2,
                text_scale=1
            )

    def _update_vehicle_tracking(self, detections: sv.Detections, frame_num: int, timestamp: float):
        """
        Update vehicle tracking states and calculate dwell times.

        Args:
            detections: Current frame detections with tracker_id
            frame_num: Current frame number
            timestamp: Current timestamp in seconds
        """
        if not self.enable_tracking or detections.tracker_id is None:
            return

        # Get current vehicles in each zone
        vehicles_in_zones: Dict[str, set] = {}
        for zone_name, zone in self.zones.items():
            mask = zone.trigger(detections)
            vehicles_in_zones[zone_name] = set(detections.tracker_id[mask])

        # Process each tracked vehicle
        for tracker_id in detections.tracker_id:
            for zone_name, vehicles_in_zone in vehicles_in_zones.items():
                is_in_zone = tracker_id in vehicles_in_zone
                zone_state = self.vehicle_states[tracker_id][zone_name]

                if is_in_zone and not zone_state:
                    # Vehicle entering zone
                    self.vehicle_states[tracker_id][zone_name] = {
                        'enter_time': frame_num,
                        'enter_timestamp': timestamp
                    }

                elif not is_in_zone and zone_state:
                    # Vehicle exiting zone - calculate dwell time
                    enter_timestamp = zone_state['enter_timestamp']
                    dwell_time = timestamp - enter_timestamp
                    self.dwell_times[zone_name].append(dwell_time)

                    # Clear state
                    self.vehicle_states[tracker_id][zone_name] = {}

    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        display: bool = False,
        save_counts: bool = True
    ) -> Dict[str, int]:
        """
        Process video and count vehicles in defined zones.

        Args:
            video_path: Path to input video file
            output_path: Path to save annotated video (optional)
            display: Whether to display video during processing
            save_counts: Whether to save count data to JSON

        Returns:
            Dictionary containing vehicle counts per zone
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

        frame_count = 0
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            vehicle_classes = [2, 3, 5, 7]
            vehicle_mask = np.isin(detections.class_id, vehicle_classes)
            detections = detections[vehicle_mask]

            # Apply tracking
            if self.enable_tracking and self.tracker:
                detections = self.tracker.update_with_detections(detections)

            # Calculate current timestamp
            timestamp = frame_count / fps

            # Update vehicle tracking and dwell times
            if self.enable_tracking:
                self._update_vehicle_tracking(detections, frame_count, timestamp)

            for zone_name, zone in self.zones.items():
                zone.trigger(detections)
                frame = self.zone_annotators[zone_name].annotate(frame)

            # Create labels with tracker IDs if tracking enabled
            if self.enable_tracking and detections.tracker_id is not None:
                labels = [
                    f"#{tracker_id} {self.model.names[class_id]} {confidence:.2f}"
                    for tracker_id, class_id, confidence in zip(
                        detections.tracker_id, detections.class_id, detections.confidence
                    )
                ]
            else:
                labels = [
                    f"{self.model.names[class_id]} {confidence:.2f}"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]

            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels)

            info_y = 30
            cv2.putText(
                frame,
                f"Frame: {frame_count}/{total_frames}",
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            for zone_name, zone in self.zones.items():
                info_y += 30
                cv2.putText(
                    frame,
                    f"{zone_name}: {int(zone.current_count)}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            if video_writer:
                video_writer.write(frame)

            if display:
                cv2.imshow("Vehicle Counting", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")

        cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()

        final_counts = {zone_name: int(zone.current_count) for zone_name, zone in self.zones.items()}

        print("\nFinal Counts:")
        for zone_name, count in final_counts.items():
            print(f"  {zone_name}: {count}")

        # Calculate dwell time statistics
        dwell_stats = {}
        if self.enable_tracking:
            print("\nDwell Time Statistics (seconds):")
            for zone_name, times in self.dwell_times.items():
                if times:
                    dwell_stats[zone_name] = {
                        'mean': float(np.mean(times)),
                        'median': float(np.median(times)),
                        'min': float(np.min(times)),
                        'max': float(np.max(times)),
                        'std': float(np.std(times)),
                        'count': len(times)
                    }
                    print(f"  {zone_name}:")
                    print(f"    Mean: {dwell_stats[zone_name]['mean']:.2f}s")
                    print(f"    Median: {dwell_stats[zone_name]['median']:.2f}s")
                    print(f"    Min: {dwell_stats[zone_name]['min']:.2f}s")
                    print(f"    Max: {dwell_stats[zone_name]['max']:.2f}s")
                    print(f"    Std Dev: {dwell_stats[zone_name]['std']:.2f}s")
                    print(f"    Vehicles tracked: {dwell_stats[zone_name]['count']}")

        if save_counts:
            counts_path = Path(video_path).with_suffix('.counts.json')
            output_data = {
                'video': str(video_path),
                'total_frames': total_frames,
                'fps': fps,
                'duration_seconds': total_frames / fps if fps > 0 else 0,
                'zones': final_counts
            }

            if self.enable_tracking and dwell_stats:
                output_data['dwell_times'] = dwell_stats

            with open(counts_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nCounts saved to: {counts_path}")
            
            # Calculate and save roundabout metrics if available
            if METRICS_AVAILABLE:
                self._calculate_roundabout_metrics(
                    video_path, final_counts, dwell_stats, 
                    total_frames, fps
                )

        return final_counts
    
    def _calculate_roundabout_metrics(self, video_path: str, zones: Dict[str, int],
                                     dwell_stats: Dict, total_frames: int, fps: int):
        """Calculate advanced roundabout traffic metrics"""
        try:
            calculator = RoundaboutMetricsCalculator()
            duration_seconds = total_frames / fps if fps > 0 else 1
            
            # Group zones by approach (north, south, east, west)
            approaches = set()
            for zone_name in zones.keys():
                parts = zone_name.split('_')
                if len(parts) >= 2:
                    approaches.add(parts[0])
            
            # Add approach data
            for approach in approaches:
                entry_count = zones.get(f'{approach}_entry', 0)
                circ_count = zones.get(f'{approach}_circulating', 0)
                exit_count = zones.get(f'{approach}_exit', 0)
                
                # Extract dwell time data
                entry_dwell_data = dwell_stats.get(f'{approach}_entry', {})
                entry_times = []
                if 'mean' in entry_dwell_data and 'count' in entry_dwell_data:
                    # Create approximate samples from mean
                    entry_times = [entry_dwell_data['mean']] * entry_dwell_data['count']
                
                calculator.add_approach_from_zone_data(
                    approach_name=approach,
                    entry_count=entry_count,
                    circulating_count=circ_count,
                    exit_count=exit_count,
                    duration_seconds=duration_seconds,
                    entry_dwell_times=entry_times
                )
            
            # Calculate overall performance
            calculator.calculate_overall_performance()
            
            # Save metrics report
            metrics_path = Path(video_path).with_suffix('.metrics.json')
            calculator.generate_report(str(metrics_path))
            
            # Print summary
            print("\n" + "=" * 70)
            print("ROUNDABOUT TRAFFIC METRICS")
            print("=" * 70)
            for name, metrics in sorted(calculator.approaches.items()):
                print(f"\n{name.upper()}:")
                print(f"  Entering Flow:    {metrics.entering_flow:6.1f} veh/h")
                print(f"  Circulating Flow: {metrics.circulating_flow:6.1f} veh/h")
                print(f"  Entry Capacity:   {metrics.entry_capacity:6.1f} veh/h")
                print(f"  Capacity Index:   {metrics.capacity_index:6.1f}%")
                print(f"  Level of Service: {metrics.level_of_service}")
            
            print(f"\nOverall LOS: {calculator.performance.overall_los}")
            print(f"Feasibility: {'✓ FEASIBLE' if calculator.performance.is_feasible else '✗ NEEDS REVIEW'}")
            print(f"\nDetailed metrics saved to: {metrics_path}")
            
        except Exception as e:
            print(f"Warning: Could not calculate roundabout metrics: {e}")

    def interactive_zone_setup(self, video_path: str, config_path: str):
        """
        Interactive tool to define polygon zones by clicking on video frame.

        Args:
            video_path: Path to video file
            config_path: Path to save configuration JSON
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read first frame from: {video_path}")

        print("\nInteractive Zone Setup - Lane Regions (In/Out)")
        print("=" * 60)
        print("Instructions:")
        print("  - Left click to add points to current zone")
        print("  - Right click to finish current zone")
        print("  - Press 's' to save configuration")
        print("  - Press 'r' to reset current zone")
        print("  - Press 'q' to quit without saving")
        print("=" * 60)
        print("\nYou will define 6 zones for directional counting:")
        print("  1. enter_in  (vehicles entering the enter lane)")
        print("  2. enter_out (vehicles exiting the enter lane)")
        print("  3. exit_in   (vehicles entering the exit lane)")
        print("  4. exit_out  (vehicles exiting the exit lane)")
        print("  5. curve_in  (vehicles entering the curve)")
        print("  6. curve_out (vehicles exiting the curve)")
        print("=" * 60)

        zones_config = {"zones": {}, "zone_colors": {}}
        current_zone = []
        zone_names = ["enter_in", "enter_out", "exit_in", "exit_out", "curve_in", "curve_out"]
        zone_colors = ["#FF0000", "#FF8800", "#00FF00", "#88FF00", "#0000FF", "#00FFFF"]
        current_zone_idx = 0

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_zone, current_zone_idx

            if event == cv2.EVENT_LBUTTONDOWN:
                current_zone.append([x, y])
                print(f"Added point ({x}, {y}) to zone '{zone_names[current_zone_idx]}'")

            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(current_zone) >= 3:
                    zone_name = zone_names[current_zone_idx]
                    zones_config["zones"][zone_name] = current_zone
                    zones_config["zone_colors"][zone_name] = zone_colors[current_zone_idx]
                    print(f"Finished zone '{zone_name}' with {len(current_zone)} points")
                    current_zone = []
                    current_zone_idx = (current_zone_idx + 1) % len(zone_names)
                else:
                    print("Need at least 3 points to define a zone")

        cv2.namedWindow("Zone Setup")
        cv2.setMouseCallback("Zone Setup", mouse_callback)

        while True:
            display_frame = frame.copy()

            for zone_name, polygon_points in zones_config["zones"].items():
                pts = np.array(polygon_points, dtype=np.int32)
                color_hex = zones_config["zone_colors"][zone_name]
                color = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
                cv2.polylines(display_frame, [pts], True, color, 2)
                cv2.fillPoly(display_frame, [pts], (*color, 50))

            if current_zone:
                pts = np.array(current_zone, dtype=np.int32)
                cv2.polylines(display_frame, [pts], False, (255, 255, 0), 2)
                for pt in current_zone:
                    cv2.circle(display_frame, tuple(pt), 5, (255, 255, 0), -1)

            cv2.putText(
                display_frame,
                f"Current zone: {zone_names[current_zone_idx]} ({len(current_zone)} points)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Zone Setup", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit without saving")
                break
            elif key == ord('s'):
                with open(config_path, 'w') as f:
                    json.dump(zones_config, f, indent=2)
                print(f"\nConfiguration saved to: {config_path}")
                break
            elif key == ord('r'):
                current_zone = []
                print("Reset current zone")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle counting for roundabout traffic analysis"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to camera configuration JSON file"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run interactive zone setup mode"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save annotated output video"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video during processing"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for detection (default: 0.3)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IOU threshold for NMS (default: 0.7)"
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable vehicle tracking and dwell time calculation"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video with detections and zones"
    )

    args = parser.parse_args()

    if args.setup:
        if not args.config:
            args.config = Path(args.video).with_suffix('.config.json')

        counter = VehicleCounter()
        counter.interactive_zone_setup(args.video, args.config)
    else:
        camera_config = None
        if args.config and Path(args.config).exists():
            with open(args.config, 'r') as f:
                camera_config = json.load(f)

        counter = VehicleCounter(
            model_path=args.model,
            camera_config=camera_config,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            enable_tracking=not args.no_tracking
        )

        # Determine output path for annotated video
        output_path = args.output
        if args.save_video and not output_path:
            # Generate default output path: video_name.annotated.mp4
            video_path = Path(args.video)
            output_path = str(video_path.with_suffix('')) + '.annotated.mp4'

        counter.process_video(
            video_path=args.video,
            output_path=output_path,
            display=args.display
        )


if __name__ == "__main__":
    main()
