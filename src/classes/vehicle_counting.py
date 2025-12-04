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
    from src.classes.roundabout_metrics import RoundaboutMetricsCalculator, RoundaboutGeometry
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
        
        # Vehicle journey tracking: {tracker_id: vehicle_info}
        self.vehicle_journeys: Dict[int, Dict] = {}
        # Track vehicle class names
        self.vehicle_classes: Dict[int, Tuple[int, str]] = {}  # {tracker_id: (class_id, class_name)}
        
        # Frame-by-frame raw detections for post-processing
        self.raw_detections_by_frame: List[Dict] = []

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

        # Update vehicle class information
        for idx, tracker_id in enumerate(detections.tracker_id):
            if tracker_id not in self.vehicle_classes:
                class_id = int(detections.class_id[idx])
                class_name = self.model.names[class_id]
                self.vehicle_classes[tracker_id] = (class_id, class_name)
                
                # Initialize journey tracking with raw data containers
                self.vehicle_journeys[tracker_id] = {
                    'tracker_id': int(tracker_id),
                    'class_id': class_id,
                    'class_name': class_name,
                    'first_seen_frame': frame_num,
                    'first_seen_time': timestamp,
                    'last_seen_frame': frame_num,
                    'last_seen_time': timestamp,
                    'zones': {},
                    'detections': []  # Frame-by-frame detection data
                }
            else:
                # Update last seen time
                self.vehicle_journeys[tracker_id]['last_seen_frame'] = frame_num
                self.vehicle_journeys[tracker_id]['last_seen_time'] = timestamp
            
            # Capture raw detection data for this vehicle at this frame
            bbox = detections.xyxy[idx]
            confidence = float(detections.confidence[idx])
            current_class_id = int(detections.class_id[idx])
            current_class_name = self.model.names[current_class_id]
            
            center_x = float((bbox[0] + bbox[2]) / 2)
            center_y = float((bbox[1] + bbox[3]) / 2)
            bbox_width = float(bbox[2] - bbox[0])
            bbox_height = float(bbox[3] - bbox[1])
            bbox_area = bbox_width * bbox_height
            
            detection_data = {
                'frame': frame_num,
                'timestamp': timestamp,
                'bbox': {
                    'x1': float(bbox[0]),
                    'y1': float(bbox[1]),
                    'x2': float(bbox[2]),
                    'y2': float(bbox[3]),
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': bbox_width,
                    'height': bbox_height,
                    'area': bbox_area
                },
                'confidence': confidence,
                'class_id': current_class_id,
                'class_name': current_class_name,
                'zones_present': []  # Will be filled below
            }
            
            # Record which zones this vehicle is in at this frame
            for zone_name, vehicles_in_zone in vehicles_in_zones.items():
                if tracker_id in vehicles_in_zone:
                    detection_data['zones_present'].append(zone_name)
            
            self.vehicle_journeys[tracker_id]['detections'].append(detection_data)
            
            # Calculate speed from previous position
            detections_list = self.vehicle_journeys[tracker_id]['detections']
            if len(detections_list) >= 2:
                prev = detections_list[-2]
                curr = detections_list[-1]
                
                dx = curr['bbox']['center_x'] - prev['bbox']['center_x']
                dy = curr['bbox']['center_y'] - prev['bbox']['center_y']
                distance = np.sqrt(dx**2 + dy**2)
                dt = curr['timestamp'] - prev['timestamp']
                
                if dt > 0:
                    speed = distance / dt
                    # Store speed data for easy access
                    if 'speeds' not in self.vehicle_journeys[tracker_id]:
                        self.vehicle_journeys[tracker_id]['speeds'] = []
                    self.vehicle_journeys[tracker_id]['speeds'].append({
                        'speed_px_per_sec': float(speed),
                        'timestamp': timestamp,
                        'frame': frame_num
                    })
            
            # Store bounding box data for easy access
            if 'bounding_boxes' not in self.vehicle_journeys[tracker_id]:
                self.vehicle_journeys[tracker_id]['bounding_boxes'] = []
            self.vehicle_journeys[tracker_id]['bounding_boxes'].append({
                'area': bbox_area,
                'width': bbox_width,
                'height': bbox_height,
                'timestamp': timestamp
            })
            
            # Store confidence data for easy access
            if 'confidence_scores' not in self.vehicle_journeys[tracker_id]:
                self.vehicle_journeys[tracker_id]['confidence_scores'] = []
            self.vehicle_journeys[tracker_id]['confidence_scores'].append({
                'confidence': confidence,
                'timestamp': timestamp
            })

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
                    
                    # Record entry in journey
                    if zone_name not in self.vehicle_journeys[tracker_id]['zones']:
                        self.vehicle_journeys[tracker_id]['zones'][zone_name] = []
                    
                    self.vehicle_journeys[tracker_id]['zones'][zone_name].append({
                        'time_entered': timestamp,
                        'frame_entered': frame_num,
                        'time_exited': None,
                        'frame_exited': None,
                        'dwell_time': None
                    })

                elif not is_in_zone and zone_state:
                    # Vehicle exiting zone - calculate dwell time
                    enter_timestamp = zone_state['enter_timestamp']
                    enter_frame = zone_state['enter_time']
                    dwell_time = timestamp - enter_timestamp
                    self.dwell_times[zone_name].append(dwell_time)
                    
                    # Update journey with exit info
                    if zone_name in self.vehicle_journeys[tracker_id]['zones']:
                        zone_visits = self.vehicle_journeys[tracker_id]['zones'][zone_name]
                        # Find the most recent entry without an exit
                        for visit in reversed(zone_visits):
                            if visit['time_exited'] is None:
                                visit['time_exited'] = timestamp
                                visit['frame_exited'] = frame_num
                                visit['dwell_time'] = dwell_time
                                break

                    # Clear state
                    self.vehicle_states[tracker_id][zone_name] = {}
    
    def _extract_metadata_from_filename(self, video_path: str) -> Dict:
        """Extract metadata from video filename (date, time, camera)"""
        import re
        from datetime import datetime
        
        filename = Path(video_path).stem
        metadata = {
            'filename': filename,
            'camera': None,
            'date': None,
            'time': None,
            'datetime': None,
            'hour': None,
            'minute': None
        }
        
        # Extract camera (e.g., normanniles1)
        camera_match = re.search(r'(normanniles\d+)', filename)
        if camera_match:
            metadata['camera'] = camera_match.group(1)
        
        # Extract datetime (format: YYYY-MM-DD-HH-MM-SS)
        datetime_match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', filename)
        if datetime_match:
            year, month, day, hour, minute, second = datetime_match.groups()
            metadata['date'] = f"{year}-{month}-{day}"
            metadata['time'] = f"{hour}:{minute}:{second}"
            metadata['datetime'] = f"{year}-{month}-{day} {hour}:{minute}:{second}"
            metadata['hour'] = int(hour)
            metadata['minute'] = int(minute)
        
        return metadata

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
            output_path = Path(video_path).with_suffix('.counts.json')
            
            # Extract metadata from filename
            metadata = self._extract_metadata_from_filename(video_path)
            
            # Create flat structure with metadata at root level
            output_data = {
                'video_path': str(video_path),
                'filename': metadata.get('filename'),
                'camera': metadata.get('camera'),
                'date': metadata.get('date'),
                'time': metadata.get('time'),
                'datetime': metadata.get('datetime'),
                'hour': metadata.get('hour'),
                'minute': metadata.get('minute'),
                'total_frames': total_frames,
                'fps': fps,
                'duration_seconds': total_frames / fps if fps > 0 else 0,
                'video_width': width,
                'video_height': height,
                'total_vehicles_tracked': 0,
            }
            
            # Add zone counts at root level
            for zone_name, count in final_counts.items():
                output_data[f'zone_count_{zone_name}'] = count
            
            if self.enable_tracking and dwell_stats:
                # Flatten dwell time stats
                for zone_name, stats in dwell_stats.items():
                    output_data[f'dwell_mean_{zone_name}'] = stats.get('mean', 0)
                    output_data[f'dwell_median_{zone_name}'] = stats.get('median', 0)
                    output_data[f'dwell_min_{zone_name}'] = stats.get('min', 0)
                    output_data[f'dwell_max_{zone_name}'] = stats.get('max', 0)
                    output_data[f'dwell_std_{zone_name}'] = stats.get('std', 0)
                    output_data[f'dwell_count_{zone_name}'] = stats.get('count', 0)
            
            # Add ML-relevant vehicle journey summaries (no raw frame-by-frame data)
            if self.enable_tracking and self.vehicle_journeys:
                output_data['vehicle_journeys'] = self._create_ml_features_from_journeys()
                output_data['total_vehicles_tracked'] = len(self.vehicle_journeys)
                
                # Add vehicle class summary (flattened)
                class_summary = defaultdict(int)
                for journey in self.vehicle_journeys.values():
                    class_summary[journey['class_name']] += 1
                for class_name, count in class_summary.items():
                    output_data[f'vehicle_class_{class_name}'] = count
                
                # Add comprehensive polygon statistics (flattened)
                polygon_stats = self._calculate_polygon_statistics()
                for zone_name, stats in polygon_stats.items():
                    output_data[f'poly_total_visits_{zone_name}'] = stats.get('total_visits', 0)
                    output_data[f'poly_unique_vehicles_{zone_name}'] = stats.get('unique_vehicles', 0)
                    output_data[f'poly_throughput_{zone_name}'] = stats.get('throughput_vehicles_per_minute', 0)
                    
                    # Vehicle types in zone
                    for vtype, count in stats.get('vehicle_types', {}).items():
                        output_data[f'poly_vtype_{vtype}_{zone_name}'] = count
                    
                    # Duration stats
                    dur_stats = stats.get('duration_stats')
                    if dur_stats:
                        output_data[f'poly_dur_mean_{zone_name}'] = dur_stats.get('mean_seconds', 0)
                        output_data[f'poly_dur_median_{zone_name}'] = dur_stats.get('median_seconds', 0)
                        output_data[f'poly_dur_std_{zone_name}'] = dur_stats.get('std_seconds', 0)
                    
                    # Speed stats
                    spd_stats = stats.get('speed_stats')
                    if spd_stats:
                        output_data[f'poly_speed_mean_{zone_name}'] = spd_stats.get('mean_px_per_sec', 0)
                        output_data[f'poly_speed_median_{zone_name}'] = spd_stats.get('median_px_per_sec', 0)
                        output_data[f'poly_speed_std_{zone_name}'] = spd_stats.get('std_px_per_sec', 0)
                    
                    # Size stats
                    size_stats = stats.get('vehicle_size_stats')
                    if size_stats:
                        output_data[f'poly_size_mean_{zone_name}'] = size_stats.get('mean_bbox_area', 0)
                        output_data[f'poly_size_median_{zone_name}'] = size_stats.get('median_bbox_area', 0)
                        output_data[f'poly_size_std_{zone_name}'] = size_stats.get('std_bbox_area', 0)
                    
                    # Confidence stats
                    conf_stats = stats.get('detection_confidence_stats')
                    if conf_stats:
                        output_data[f'poly_conf_mean_{zone_name}'] = conf_stats.get('mean_confidence', 0)
                        output_data[f'poly_conf_median_{zone_name}'] = conf_stats.get('median_confidence', 0)

            # Calculate and add roundabout metrics if available (flattened)
            if METRICS_AVAILABLE:
                roundabout_metrics = self._calculate_roundabout_metrics(
                    video_path, final_counts, dwell_stats, 
                    total_frames, fps
                )
                if roundabout_metrics:
                    # Flatten roundabout metrics
                    overall = roundabout_metrics.get('overall_performance', {})
                    output_data['roundabout_total_entering_flow'] = overall.get('total_entering_flow', 0)
                    output_data['roundabout_total_circulating_flow'] = overall.get('total_circulating_flow', 0)
                    output_data['roundabout_overall_capacity_index'] = overall.get('overall_capacity_index', 0)
                    output_data['roundabout_is_feasible'] = overall.get('is_feasible', True)
                    output_data['roundabout_overall_los'] = overall.get('overall_los', 'Unknown')
                    
                    # Flatten approach-specific metrics
                    for approach_name, approach_data in roundabout_metrics.get('approaches', {}).items():
                        output_data[f'roundabout_{approach_name}_entering_flow'] = approach_data.get('entering_flow', 0)
                        output_data[f'roundabout_{approach_name}_circulating_flow'] = approach_data.get('circulating_flow', 0)
                        output_data[f'roundabout_{approach_name}_entry_capacity'] = approach_data.get('entry_capacity', 0)
                        output_data[f'roundabout_{approach_name}_capacity_index'] = approach_data.get('capacity_index', 0)
                        output_data[f'roundabout_{approach_name}_level_of_service'] = approach_data.get('level_of_service', 'Unknown')
                        output_data[f'roundabout_{approach_name}_average_delay'] = approach_data.get('average_delay', 0)
                        output_data[f'roundabout_{approach_name}_queue_length'] = approach_data.get('queue_length', 0)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nAll data saved to: {output_path}")
            
            if self.enable_tracking and self.vehicle_journeys:
                print(f"\nVehicle Tracking Summary:")
                print(f"  Total vehicles tracked: {output_data['total_vehicles_tracked']}")
                vehicle_classes = {k.replace('vehicle_class_', ''): v for k, v in output_data.items() if k.startswith('vehicle_class_')}
                if vehicle_classes:
                    print(f"  Vehicle classes:")
                    for class_name, count in sorted(vehicle_classes.items()):
                        print(f"    {class_name}: {count}")

        return final_counts
    
    def _calculate_polygon_statistics(self) -> Dict:
        """Calculate comprehensive statistics for each polygon/zone"""
        polygon_stats = {}
        
        for zone_name in self.zones.keys():
            stats = {
                'total_vehicles': 0,
                'vehicles_by_class': defaultdict(int),
                'entry_exit_durations': [],
                'speeds': [],
                'vehicle_ids': [],
                'first_entry_time': None,
                'last_exit_time': None,
                'avg_bbox_area': [],
                'confidence_scores': []
            }
            
            # Collect data from all vehicle journeys
            for journey in self.vehicle_journeys.values():
                if zone_name in journey.get('zones', {}):
                    visits = journey['zones'][zone_name]
                    
                    for visit in visits:
                        stats['total_vehicles'] += 1
                        stats['vehicles_by_class'][journey['class_name']] += 1
                        stats['vehicle_ids'].append(journey['tracker_id'])
                        
                        # Entry/exit duration
                        if visit.get('dwell_time') is not None:
                            stats['entry_exit_durations'].append(visit['dwell_time'])
                        
                        # Track first entry and last exit
                        if visit.get('time_entered') is not None:
                            if stats['first_entry_time'] is None or visit['time_entered'] < stats['first_entry_time']:
                                stats['first_entry_time'] = visit['time_entered']
                        
                        if visit.get('time_exited') is not None:
                            if stats['last_exit_time'] is None or visit['time_exited'] > stats['last_exit_time']:
                                stats['last_exit_time'] = visit['time_exited']
                        
                        # Calculate speed in this zone (from trajectory)
                        zone_speeds = self._get_speeds_in_zone(
                            journey, 
                            visit.get('frame_entered'), 
                            visit.get('frame_exited')
                        )
                        stats['speeds'].extend(zone_speeds)
                        
                        # Get average bbox area in this zone
                        zone_bbox_areas = self._get_bbox_areas_in_zone(
                            journey,
                            visit.get('time_entered'),
                            visit.get('time_exited')
                        )
                        stats['avg_bbox_area'].extend(zone_bbox_areas)
                        
                        # Get confidence scores in this zone
                        zone_confidences = self._get_confidences_in_zone(
                            journey,
                            visit.get('time_entered'),
                            visit.get('time_exited')
                        )
                        stats['confidence_scores'].extend(zone_confidences)
            
            # Calculate summary statistics
            summary = {
                'total_visits': stats['total_vehicles'],  # Total entries (a vehicle can visit multiple times)
                'unique_vehicles': len(set(stats['vehicle_ids'])),  # Unique vehicles that visited
                'vehicle_types': dict(stats['vehicles_by_class']),  # Count by vehicle class
                'vehicle_type_percentages': {}  # Percentage breakdown
            }
            
            # Calculate percentage breakdown of vehicle types
            if stats['total_vehicles'] > 0:
                for class_name, count in stats['vehicles_by_class'].items():
                    summary['vehicle_type_percentages'][class_name] = round(
                        (count / stats['total_vehicles']) * 100, 2
                    )
            
            # Add raw list of vehicle IDs for reference
            summary['vehicle_ids_list'] = sorted(set(stats['vehicle_ids']))
            
            # Duration statistics
            if stats['entry_exit_durations']:
                summary['duration_stats'] = {
                    'mean_seconds': float(np.mean(stats['entry_exit_durations'])),
                    'median_seconds': float(np.median(stats['entry_exit_durations'])),
                    'min_seconds': float(np.min(stats['entry_exit_durations'])),
                    'max_seconds': float(np.max(stats['entry_exit_durations'])),
                    'std_seconds': float(np.std(stats['entry_exit_durations'])),
                    'total_samples': len(stats['entry_exit_durations'])
                }
            else:
                summary['duration_stats'] = None
            
            # Speed statistics (pixels per second)
            if stats['speeds']:
                summary['speed_stats'] = {
                    'mean_px_per_sec': float(np.mean(stats['speeds'])),
                    'median_px_per_sec': float(np.median(stats['speeds'])),
                    'min_px_per_sec': float(np.min(stats['speeds'])),
                    'max_px_per_sec': float(np.max(stats['speeds'])),
                    'std_px_per_sec': float(np.std(stats['speeds'])),
                    'total_samples': len(stats['speeds'])
                }
            else:
                summary['speed_stats'] = None
            
            # Bounding box area statistics (proxy for vehicle size)
            if stats['avg_bbox_area']:
                summary['vehicle_size_stats'] = {
                    'mean_bbox_area': float(np.mean(stats['avg_bbox_area'])),
                    'median_bbox_area': float(np.median(stats['avg_bbox_area'])),
                    'std_bbox_area': float(np.std(stats['avg_bbox_area'])),
                    'total_samples': len(stats['avg_bbox_area'])
                }
            else:
                summary['vehicle_size_stats'] = None
            
            # Confidence statistics
            if stats['confidence_scores']:
                summary['detection_confidence_stats'] = {
                    'mean_confidence': float(np.mean(stats['confidence_scores'])),
                    'median_confidence': float(np.median(stats['confidence_scores'])),
                    'min_confidence': float(np.min(stats['confidence_scores'])),
                    'std_confidence': float(np.std(stats['confidence_scores'])),
                    'total_samples': len(stats['confidence_scores'])
                }
            else:
                summary['detection_confidence_stats'] = None
            
            # Time range
            if stats['first_entry_time'] is not None and stats['last_exit_time'] is not None:
                summary['time_range'] = {
                    'first_entry_time': float(stats['first_entry_time']),
                    'last_exit_time': float(stats['last_exit_time']),
                    'active_duration': float(stats['last_exit_time'] - stats['first_entry_time'])
                }
            else:
                summary['time_range'] = None
            
            # Calculate throughput (vehicles per minute)
            if summary.get('time_range') and summary['time_range']['active_duration'] > 0:
                summary['throughput_vehicles_per_minute'] = (
                    stats['total_vehicles'] / (summary['time_range']['active_duration'] / 60.0)
                )
            else:
                summary['throughput_vehicles_per_minute'] = None
            
            polygon_stats[zone_name] = summary
        
        return polygon_stats
    
    def _get_speeds_in_zone(self, journey: Dict, frame_start: Optional[int], frame_end: Optional[int]) -> List[float]:
        """Extract speeds from trajectory within a zone's time range"""
        speeds = []
        
        if frame_start is None or frame_end is None:
            return speeds
        
        for speed_data in journey.get('speeds', []):
            # Check if this speed measurement falls within the zone visit
            if frame_start <= speed_data.get('frame', 0) <= frame_end:
                speeds.append(speed_data['speed_px_per_sec'])
        
        return speeds
    
    def _get_bbox_areas_in_zone(self, journey: Dict, time_start: Optional[float], time_end: Optional[float]) -> List[float]:
        """Extract bounding box areas within a zone's time range"""
        areas = []
        
        if time_start is None or time_end is None:
            return areas
        
        for bbox_data in journey.get('bounding_boxes', []):
            if time_start <= bbox_data.get('timestamp', 0) <= time_end:
                areas.append(bbox_data['area'])
        
        return areas
    
    def _get_confidences_in_zone(self, journey: Dict, time_start: Optional[float], time_end: Optional[float]) -> List[float]:
        """Extract confidence scores within a zone's time range"""
        confidences = []
        
        if time_start is None or time_end is None:
            return confidences
        
        for conf_data in journey.get('confidence_scores', []):
            if time_start <= conf_data.get('timestamp', 0) <= time_end:
                confidences.append(conf_data['confidence'])
        
        return confidences
    
    def _create_ml_features_from_journeys(self) -> List[Dict]:
        """Create ML-ready features from vehicle journeys (no raw frame data)"""
        ml_features = []
        
        for journey in self.vehicle_journeys.values():
            # Calculate trajectory-based features
            detections = journey.get('detections', [])
            speeds = journey.get('speeds', [])
            bboxes = journey.get('bounding_boxes', [])
            confidences = journey.get('confidence_scores', [])
            
            # Aggregate features
            feature = {
                'vehicle_id': journey['tracker_id'],
                'class_id': journey['class_id'],
                'class_name': journey['class_name'],
                
                # Temporal features
                'first_seen_time': journey['first_seen_time'],
                'last_seen_time': journey['last_seen_time'],
                'total_time_visible': journey['last_seen_time'] - journey['first_seen_time'],
                'total_frames_tracked': len(detections),
                
                # Zone interaction features
                'zones_visited': list(journey.get('zones', {}).keys()),
                'num_zones_visited': len(journey.get('zones', {})),
                'zone_sequences': self._extract_zone_sequence(journey),
                
                # Per-zone dwell times
                'zone_dwell_times': {
                    zone: [v['dwell_time'] for v in visits if v['dwell_time'] is not None]
                    for zone, visits in journey.get('zones', {}).items()
                },
                
                # Movement features (aggregated)
                'speed_stats': {
                    'mean': float(np.mean([s['speed_px_per_sec'] for s in speeds])) if speeds else 0,
                    'max': float(np.max([s['speed_px_per_sec'] for s in speeds])) if speeds else 0,
                    'min': float(np.min([s['speed_px_per_sec'] for s in speeds])) if speeds else 0,
                    'std': float(np.std([s['speed_px_per_sec'] for s in speeds])) if speeds else 0
                },
                
                # Size features (aggregated)
                'size_stats': {
                    'mean_area': float(np.mean([b['area'] for b in bboxes])) if bboxes else 0,
                    'mean_width': float(np.mean([b['width'] for b in bboxes])) if bboxes else 0,
                    'mean_height': float(np.mean([b['height'] for b in bboxes])) if bboxes else 0
                },
                
                # Detection quality
                'detection_quality': {
                    'mean_confidence': float(np.mean([c['confidence'] for c in confidences])) if confidences else 0,
                    'min_confidence': float(np.min([c['confidence'] for c in confidences])) if confidences else 0
                },
                
                # Trajectory summary (start/end positions only)
                'trajectory_summary': {
                    'start_position': detections[0]['bbox'] if detections else None,
                    'end_position': detections[-1]['bbox'] if detections else None,
                    'total_distance': self._calculate_total_distance(detections)
                }
            }
            
            ml_features.append(feature)
        
        return ml_features
    
    def _extract_zone_sequence(self, journey: Dict) -> List[Dict]:
        """Extract ordered sequence of zone visits with timing"""
        sequence = []
        for zone_name, visits in journey.get('zones', {}).items():
            for visit in visits:
                sequence.append({
                    'zone': zone_name,
                    'entered': visit['time_entered'],
                    'exited': visit.get('time_exited'),
                    'duration': visit.get('dwell_time')
                })
        # Sort by entry time
        return sorted(sequence, key=lambda x: x['entered'] if x['entered'] else float('inf'))
    
    def _calculate_total_distance(self, detections: List[Dict]) -> float:
        """Calculate total distance traveled from detections"""
        if len(detections) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(detections)):
            prev = detections[i-1]['bbox']
            curr = detections[i]['bbox']
            dx = curr['center_x'] - prev['center_x']
            dy = curr['center_y'] - prev['center_y']
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return float(total_distance)
    
    def _calculate_roundabout_metrics(self, video_path: str, zones: Dict[str, int],
                                     dwell_stats: Dict, total_frames: int, fps: int) -> Optional[Dict]:
        """Calculate advanced roundabout traffic metrics and return as dict"""
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
            
            # Generate metrics as dictionary (don't save separate file)
            from dataclasses import asdict
            metrics_dict = {
                'geometry': asdict(calculator.geometry),
                'approaches': {
                    name: asdict(metrics) 
                    for name, metrics in calculator.approaches.items()
                },
                'overall_performance': asdict(calculator.performance),
                'summary': {
                    'total_approaches': len(calculator.approaches),
                    'critical_approach': calculator.performance.worst_approach_name,
                    'overall_los': calculator.performance.overall_los,
                    'is_feasible': calculator.performance.is_feasible,
                    'total_entering_vehicles': sum(m.total_vehicles for m in calculator.approaches.values()),
                }
            }
            
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
            
            return metrics_dict
            
        except Exception as e:
            print(f"Warning: Could not calculate roundabout metrics: {e}")
            return None

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
