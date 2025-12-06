#!/usr/bin/env python3
"""
Advanced Object Tracking with Zone-Based Journey Validation

This module implements YOLO + ByteTrack for tracking vehicles through roundabout zones.
Each vehicle must complete a valid journey: enter zone_start -> exit zone_start -> enter zone_end -> exit zone_end
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ZoneEvent:
    """Represents a single zone event (entry or exit)"""
    def __init__(self, event_type: str, zone_name: str, frame: int, timestamp: float):
        self.event_type = event_type  # 'enter' or 'exit'
        self.zone_name = zone_name
        self.frame = frame
        self.timestamp = timestamp
    
    def __repr__(self):
        return f"ZoneEvent({self.event_type}, {self.zone_name}, f={self.frame}, t={self.timestamp:.2f}s)"


class VehicleJourney:
    """Tracks a single vehicle's complete journey through zones"""
    def __init__(self, tracker_id: int, class_id: int, class_name: str):
        self.tracker_id = tracker_id
        self.class_id = class_id
        self.class_name = class_name
        self.events: List[ZoneEvent] = []
        self.current_zones: set = set()  # Zones the vehicle is currently in
        self.first_seen_frame = None
        self.last_seen_frame = None
        self.first_seen_time = None
        self.last_seen_time = None
    
    def add_event(self, event: ZoneEvent):
        """Add a zone event to the journey"""
        self.events.append(event)
    
    def is_valid_journey(self, entry_zone: str, exit_zone: str, partial_ok: bool = True) -> bool:
        """
        Validate that vehicle completed a journey through zones.
        
        Full journey (4 events):
        1. t1_enter: Enter entry zone
        2. t2_exit: Exit entry zone
        3. t3_enter: Enter exit zone
        4. t4_exit: Exit exit zone
        
        Partial journey (3 events) - allowed if partial_ok=True:
        1. t1_enter: Enter entry zone
        2. t2_exit: Exit entry zone
        3. t3_enter: Enter exit zone
        (Vehicle may have left frame or tracking lost before final exit)
        
        Args:
            entry_zone: Name of the start zone (e.g., 'enter')
            exit_zone: Name of the end zone (e.g., 'circulating_right')
            partial_ok: Allow partial journeys (default True)
        
        Returns:
            True if journey is valid (full or partial)
        """
        # Extract events for the relevant zones
        entry_events = [e for e in self.events if e.zone_name == entry_zone]
        exit_events = [e for e in self.events if e.zone_name == exit_zone]
        
        # Check for required sequence
        has_t1_enter = any(e.event_type == 'enter' for e in entry_events)
        has_t2_exit = any(e.event_type == 'exit' for e in entry_events)
        has_t3_enter = any(e.event_type == 'enter' for e in exit_events)
        has_t4_exit = any(e.event_type == 'exit' for e in exit_events)
        
        # Full journey validation (4 events)
        if has_t1_enter and has_t2_exit and has_t3_enter and has_t4_exit:
            t1 = next((e for e in entry_events if e.event_type == 'enter'), None)
            t2 = next((e for e in entry_events if e.event_type == 'exit'), None)
            t3 = next((e for e in exit_events if e.event_type == 'enter'), None)
            t4 = next((e for e in exit_events if e.event_type == 'exit'), None)
            
            # Check timestamps are in order: t1 < t2 < t3 < t4
            if t1 and t2 and t3 and t4:
                return (t1.timestamp < t2.timestamp < t3.timestamp < t4.timestamp)
        
        # Partial journey validation (3 events) - vehicle entered end zone but didn't exit
        if partial_ok and has_t1_enter and has_t2_exit and has_t3_enter:
            t1 = next((e for e in entry_events if e.event_type == 'enter'), None)
            t2 = next((e for e in entry_events if e.event_type == 'exit'), None)
            t3 = next((e for e in exit_events if e.event_type == 'enter'), None)
            
            # Check timestamps are in order: t1 < t2 < t3
            if t1 and t2 and t3:
                return (t1.timestamp < t2.timestamp < t3.timestamp)
        
        return False
    
    def get_journey_times(self, entry_zone: str, exit_zone: str, zone_distances: Optional[Dict] = None) -> Optional[Dict[str, float]]:
        """
        Get the key timestamps for a valid journey (full or partial)

        Args:
            entry_zone: Name of entry zone
            exit_zone: Name of exit zone
            zone_distances: Optional dict with distance info in meters

        Returns:
            Dict with timestamps, journey type, and speeds (if distances provided), or None if invalid
        """
        if not self.is_valid_journey(entry_zone, exit_zone):
            return None

        entry_events = [e for e in self.events if e.zone_name == entry_zone]
        exit_events = [e for e in self.events if e.zone_name == exit_zone]

        t1 = next((e for e in entry_events if e.event_type == 'enter'), None)
        t2 = next((e for e in entry_events if e.event_type == 'exit'), None)
        t3 = next((e for e in exit_events if e.event_type == 'enter'), None)
        t4 = next((e for e in exit_events if e.event_type == 'exit'), None)

        # Determine if full or partial journey
        is_complete = t4 is not None

        result = {
            't1_enter_start_zone': t1.timestamp,
            't2_exit_start_zone': t2.timestamp,
            't3_enter_end_zone': t3.timestamp,
            'journey_type': 'complete' if is_complete else 'partial',
            'time_in_start_zone': t2.timestamp - t1.timestamp,
            'time_between_zones': t3.timestamp - t2.timestamp,
        }

        if is_complete:
            result['t4_exit_end_zone'] = t4.timestamp
            result['time_in_end_zone'] = t4.timestamp - t3.timestamp
            result['total_journey_time'] = t4.timestamp - t1.timestamp
        else:
            result['t4_exit_end_zone'] = None
            result['time_in_end_zone'] = None
            result['total_journey_time'] = t3.timestamp - t1.timestamp

        # Calculate speeds if distances are provided
        if zone_distances:
            # Look for distance between entry and exit zones
            distance_key = f"{entry_zone}_to_{exit_zone}"
            reverse_key = f"{exit_zone}_to_{entry_zone}"

            distance_meters = None
            if distance_key in zone_distances:
                distance_meters = zone_distances[distance_key]
            elif reverse_key in zone_distances:
                distance_meters = zone_distances[reverse_key]

            if distance_meters and result['total_journey_time'] > 0:
                # Speed in m/s
                speed_ms = distance_meters / result['total_journey_time']
                # Speed in km/h
                speed_kmh = speed_ms * 3.6

                result['distance_meters'] = distance_meters
                result['speed_m_s'] = speed_ms
                result['speed_km_h'] = speed_kmh

        return result
    
    def to_dict(self) -> Dict:
        """Convert journey to dictionary for JSON export"""
        return {
            'tracker_id': self.tracker_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'first_seen_frame': self.first_seen_frame,
            'last_seen_frame': self.last_seen_frame,
            'first_seen_time': self.first_seen_time,
            'last_seen_time': self.last_seen_time,
            'events': [
                {
                    'event_type': e.event_type,
                    'zone_name': e.zone_name,
                    'frame': e.frame,
                    'timestamp': e.timestamp
                } for e in self.events
            ]
        }


class RoundaboutTracker:
    """Main tracker for roundabout vehicle analysis with zone validation"""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        zone_distances: Optional[Dict] = None,
        log_level: str = "INFO"
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Load YOLO model
        self.logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        # Tracking configuration
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tracker = sv.ByteTrack()

        # Zone management
        self.zones: Dict[str, sv.PolygonZone] = {}
        self.zone_annotators: Dict[str, sv.PolygonZoneAnnotator] = {}

        # Distance configuration for speed calculation
        self.zone_distances = zone_distances or {}
        if self.zone_distances:
            self.logger.info(f"Loaded {len(self.zone_distances)} zone distance measurements")

        # Journey tracking
        self.vehicle_journeys: Dict[int, VehicleJourney] = {}

        # Track which zones each vehicle was in on the previous frame
        self.previous_zone_states: Dict[int, set] = defaultdict(set)

        self.logger.info("RoundaboutTracker initialized successfully")
    
    def load_zones_from_config(self, camera_config_path: str):
        """
        Load polygon zones from camera configuration JSON
        
        Args:
            camera_config_path: Path to camera_X_zones.json file
        """
        with open(camera_config_path, 'r') as f:
            config = json.load(f)
        
        zones_config = config.get("zones", {})
        zone_colors = config.get("zone_colors", {})
        
        self.logger.info(f"Loading {len(zones_config)} zones from {camera_config_path}")
        
        for zone_name, polygon_points in zones_config.items():
            polygon = np.array(polygon_points, dtype=np.int32)
            
            # Create zone
            self.zones[zone_name] = sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=[sv.Position.BOTTOM_CENTER]
            )
            
            # Create annotator
            color_hex = zone_colors.get(zone_name, "#FF0000")
            self.zone_annotators[zone_name] = sv.PolygonZoneAnnotator(
                zone=self.zones[zone_name],
                color=sv.Color.from_hex(color_hex),
                thickness=2,
                text_thickness=2,
                text_scale=0.8
            )
            
            self.logger.info(f"  Zone '{zone_name}': {len(polygon_points)} points")
    
    def _detect_zone_events(
        self,
        detections: sv.Detections,
        frame_num: int,
        timestamp: float
    ):
        """
        Detect zone entry/exit events for all tracked vehicles
        
        Args:
            detections: Current frame detections with tracker_id
            frame_num: Current frame number
            timestamp: Timestamp in seconds
        """
        if detections.tracker_id is None:
            return
        
        # Get current zone states for all vehicles
        current_zone_states: Dict[int, set] = defaultdict(set)
        
        for zone_name, zone in self.zones.items():
            mask = zone.trigger(detections)
            vehicles_in_zone = detections.tracker_id[mask]
            
            for tracker_id in vehicles_in_zone:
                current_zone_states[int(tracker_id)].add(zone_name)
        
        # Compare with previous states to detect entry/exit events
        all_tracker_ids = set(detections.tracker_id)
        
        for tracker_id in all_tracker_ids:
            tracker_id = int(tracker_id)
            current_zones = current_zone_states.get(tracker_id, set())
            previous_zones = self.previous_zone_states.get(tracker_id, set())
            
            # Detect entries (in current but not in previous)
            entered_zones = current_zones - previous_zones
            for zone_name in entered_zones:
                event = ZoneEvent('enter', zone_name, frame_num, timestamp)
                self.vehicle_journeys[tracker_id].add_event(event)
                self.logger.debug(f"Vehicle {tracker_id} entered {zone_name} at {timestamp:.2f}s")
            
            # Detect exits (in previous but not in current)
            exited_zones = previous_zones - current_zones
            for zone_name in exited_zones:
                event = ZoneEvent('exit', zone_name, frame_num, timestamp)
                self.vehicle_journeys[tracker_id].add_event(event)
                self.logger.debug(f"Vehicle {tracker_id} exited {zone_name} at {timestamp:.2f}s")
            
            # Update zone state
            self.previous_zone_states[tracker_id] = current_zones
    
    def _update_vehicle_journeys(
        self,
        detections: sv.Detections,
        frame_num: int,
        timestamp: float
    ):
        """
        Update journey information for all detected vehicles
        
        Args:
            detections: Current frame detections with tracker_id
            frame_num: Current frame number
            timestamp: Timestamp in seconds
        """
        if detections.tracker_id is None:
            return
        
        for idx, tracker_id in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id)
            
            # Initialize journey if new vehicle
            if tracker_id not in self.vehicle_journeys:
                class_id = int(detections.class_id[idx])
                class_name = self.model.names[class_id]
                
                journey = VehicleJourney(tracker_id, class_id, class_name)
                journey.first_seen_frame = frame_num
                journey.first_seen_time = timestamp
                
                self.vehicle_journeys[tracker_id] = journey
                self.logger.debug(f"New vehicle {tracker_id} ({class_name}) detected at {timestamp:.2f}s")
            
            # Update last seen
            self.vehicle_journeys[tracker_id].last_seen_frame = frame_num
            self.vehicle_journeys[tracker_id].last_seen_time = timestamp
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = False,
        save_json: bool = True
    ) -> Dict:
        """
        Process video and track vehicles through zones
        
        Args:
            video_path: Path to input video file
            output_path: Path to output annotated video (optional)
            show_preview: Whether to display preview window
            save_json: Whether to save journey data to JSON
        
        Returns:
            Dictionary containing tracking results and statistics
        """
        video_path = Path(video_path)
        
        # Extract camera number from filename (e.g., normanniles1_xxx.mp4 -> camera 1)
        camera_match = re.search(r'normanniles(\d+)', video_path.stem)
        if not camera_match:
            raise ValueError(f"Cannot extract camera number from filename: {video_path.name}")
        
        camera_num = int(camera_match.group(1))
        config_path = Path(__file__).parent / "camera_configs" / f"camera_{camera_num}_zones.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Camera config not found: {config_path}")
        
        self.logger.info(f"Processing video: {video_path.name}")
        self.logger.info(f"Using config: {config_path.name}")
        
        # Load zones
        self.load_zones_from_config(str(config_path))
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
        
        # Setup video writer if needed
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try different codecs for better compatibility
            # First try H264 (x264) which is widely supported
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Check if writer opened successfully
            if not writer.isOpened():
                self.logger.warning("H.264 codec failed, trying mp4v...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not writer.isOpened():
                self.logger.warning("mp4v codec failed, trying XVID...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # Change extension to .avi for XVID
                output_path = output_path.with_suffix('.avi')
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if writer.isOpened():
                self.logger.info(f"Output video: {output_path}")
            else:
                self.logger.error("Failed to initialize video writer with any codec")
                writer = None
        
        # Process frames
        frame_num = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_num / fps
                
                # Run detection
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )[0]
                
                # Convert to supervision format
                detections = sv.Detections.from_ultralytics(results)
                
                # Apply tracking
                detections = self.tracker.update_with_detections(detections)
                
                # Update vehicle journeys
                self._update_vehicle_journeys(detections, frame_num, timestamp)
                
                # Detect zone events (entry/exit)
                self._detect_zone_events(detections, frame_num, timestamp)
                
                # Annotate frame
                annotated_frame = frame.copy()
                
                # Draw zones
                for zone_name, zone_annotator in self.zone_annotators.items():
                    annotated_frame = zone_annotator.annotate(annotated_frame)
                
                # Draw detections with tracker IDs
                box_annotator = sv.BoxAnnotator(thickness=2)
                label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
                
                labels = []
                if detections.tracker_id is not None:
                    for idx, tracker_id in enumerate(detections.tracker_id):
                        class_id = int(detections.class_id[idx])
                        class_name = self.model.names[class_id]
                        confidence = detections.confidence[idx]
                        labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")
                
                annotated_frame = box_annotator.annotate(annotated_frame, detections)
                if labels:
                    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
                
                # Add frame info
                info_text = f"Frame: {frame_num}/{total_frames} | Time: {timestamp:.2f}s | Vehicles: {len(self.vehicle_journeys)}"
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Write/show frame
                if writer:
                    writer.write(annotated_frame)
                
                if show_preview:
                    cv2.imshow('Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_num += 1
                
                if frame_num % 100 == 0:
                    self.logger.info(f"Processed {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        self.logger.info(f"Processing complete. Tracked {len(self.vehicle_journeys)} vehicles")
        
        # Compile results
        results = self._compile_results(video_path.stem)
        
        # Save JSON if requested
        if save_json:
            json_path = video_path.parent / f"{video_path.stem}_tracking.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved tracking data: {json_path}")
        
        return results
    
    def _compile_results(self, video_name: str) -> Dict:
        """
        Compile tracking results with validation statistics
        
        Valid journey patterns:
        1. circulating_left -> circulating_right
        2. circulating_left -> exit
        3. enter -> circulating_right
        
        Each pattern accepts both complete (4 events) and partial (3 events) journeys.
        
        Args:
            video_name: Name of the processed video
        
        Returns:
            Dictionary with complete tracking results
        """
        zone_names = list(self.zones.keys())
        
        # Define valid journey patterns
        valid_patterns = [
            ('circulating_left', 'circulating_right'),
            ('circulating_left', 'exit'),
            ('enter', 'circulating_right')
        ]
        
        results = {
            'video_name': video_name,
            'total_vehicles_detected': len(self.vehicle_journeys),
            'zones': zone_names,
            'valid_patterns': [f"{s} -> {e}" for s, e in valid_patterns],
            'vehicles': []
        }
        
        # Validate each vehicle journey
        valid_count = 0
        for tracker_id, journey in self.vehicle_journeys.items():
            vehicle_data = journey.to_dict()
            
            # Check validation for each valid pattern
            validations = []
            for start_zone, end_zone in valid_patterns:
                is_valid = journey.is_valid_journey(start_zone, end_zone)
                journey_times = journey.get_journey_times(start_zone, end_zone, self.zone_distances)

                validations.append({
                    'pattern': f"{start_zone} -> {end_zone}",
                    'start_zone': start_zone,
                    'end_zone': end_zone,
                    'is_valid': is_valid,
                    'journey_times': journey_times
                })

                if is_valid:
                    valid_count += 1
            
            vehicle_data['validations'] = validations
            vehicle_data['has_valid_journey'] = any(v['is_valid'] for v in validations)
            
            results['vehicles'].append(vehicle_data)
        
        results['valid_journeys_count'] = valid_count
        results['validation_rate'] = valid_count / len(self.vehicle_journeys) if self.vehicle_journeys else 0
        
        self.logger.info(f"Valid journeys: {valid_count}/{len(self.vehicle_journeys)} ({results['validation_rate']*100:.1f}%)")
        
        return results


def track_video_with_zones(
    video_path: str,
    model_path: str = "yolov8n.pt",
    output_video: Optional[str] = None,
    confidence: float = 0.3,
    zone_distances_json: Optional[str] = None,
    show_preview: bool = False
) -> Dict:
    """
    Main function to track vehicles in a video with zone validation

    This function:
    1. Extracts camera number from video filename (e.g., normanniles1_xxx.mp4)
    2. Loads corresponding zone polygons from camera_configs/camera_X_zones.json
       Zones: enter, exit, circulating_left, circulating_right
    3. Tracks vehicles using YOLO + ByteTrack
    4. Validates vehicles against three valid journey patterns:
       - Pattern 1: circulating_left -> circulating_right
       - Pattern 2: circulating_left -> exit
       - Pattern 3: enter -> circulating_right
    5. Each journey requires 4 timestamps:
       - t1_enter: Vehicle enters start zone
       - t2_exit: Vehicle exits start zone
       - t3_enter: Vehicle enters end zone
       - t4_exit: Vehicle exits end zone
    6. Calculates speed if zone_distances_json is provided
    7. Returns detailed tracking results with validation status

    Args:
        video_path: Path to video file (must contain 'normanniles[1-4]' in filename)
        model_path: Path to YOLO model weights
        output_video: Optional path to save annotated video
        confidence: Detection confidence threshold
        zone_distances_json: Optional path to JSON file with zone distances in meters
            Format: {"circulating_left_to_circulating_right": 18.5,
                     "circulating_left_to_exit": 22.0,
                     "enter_to_circulating_right": 20.0}
        show_preview: Whether to show live preview

    Returns:
        Dictionary containing:
        - All tracked vehicles with their journeys
        - Zone entry/exit events with timestamps
        - Validation status for each vehicle against the 3 patterns
        - Journey statistics
        - Speed calculations (if distances provided)

    Example:
        >>> results = track_video_with_zones(
        ...     "normanniles1_2025-10-20-06-00-45.mp4",
        ...     output_video="output.mp4",
        ...     zone_distances_json="camera_configs/zone_distances.json"
        ... )
        >>> print(f"Valid journeys: {results['valid_journeys_count']}")
        >>> print(f"Valid patterns: {results['valid_patterns']}")
    """
    # Load zone distances if provided
    zone_distances = None
    if zone_distances_json:
        zone_distances_path = Path(zone_distances_json)
        if zone_distances_path.exists():
            with open(zone_distances_path, 'r') as f:
                zone_distances = json.load(f)
            logger.info(f"Loaded zone distances from {zone_distances_json}")
        else:
            logger.warning(f"Zone distances file not found: {zone_distances_json}")

    tracker = RoundaboutTracker(
        model_path=model_path,
        confidence_threshold=confidence,
        zone_distances=zone_distances,
        log_level="INFO"
    )

    results = tracker.process_video(
        video_path=video_path,
        output_path=output_video,
        show_preview=show_preview,
        save_json=True
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track vehicles with zone validation")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--zone-distances", type=str, help="Path to JSON file with zone distances in meters")
    parser.add_argument("--preview", action="store_true", help="Show live preview")

    args = parser.parse_args()

    results = track_video_with_zones(
        video_path=args.video_path,
        model_path=args.model,
        output_video=args.output,
        confidence=args.confidence,
        zone_distances_json=args.zone_distances,
        show_preview=args.preview
    )

    print("\n" + "="*80)
    print("TRACKING SUMMARY")
    print("="*80)
    print(f"Video: {results['video_name']}")
    print(f"Total vehicles detected: {results['total_vehicles_detected']}")
    print(f"Valid journeys: {results['valid_journeys_count']}")
    print(f"Validation rate: {results['validation_rate']*100:.1f}%")
    print(f"Zones: {', '.join(results['zones'])}")
    print("="*80)
