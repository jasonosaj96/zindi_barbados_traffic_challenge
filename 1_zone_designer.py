#!/usr/bin/env python3
"""
Interactive Zone Designer for Roundabout Traffic Analysis

This tool allows you to design detection zones with multiple directions:
- North, South, East, West
- Each direction has: entry, exit, queue, and circulating zones
- Total of 16 zones for comprehensive roundabout analysis
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse


class ZoneDesigner:
    """Interactive tool to design detection zones for traffic analysis"""

    def __init__(self, video_path, config_path=None, direction=None):
        self.video_path = video_path
        self.config_path = config_path
        self.cap = cv2.VideoCapture(video_path)
        self.zones = {}
        self.current_zone_name = None
        self.current_points = []
        self.direction = direction

        # Define all zone types based on direction
        if direction:
            # Only zones for the specified direction
            self.zone_types = [
                f'{direction}_entry',
                f'{direction}_exit',
                f'{direction}_queue',
                f'{direction}_circulating'
            ]
        else:
            # All 16 zones (legacy mode)
            self.zone_types = [
                'north_entry', 'north_exit', 'north_queue', 'north_circulating',
                'south_entry', 'south_exit', 'south_queue', 'south_circulating',
                'east_entry', 'east_exit', 'east_queue', 'east_circulating',
                'west_entry', 'west_exit', 'west_queue', 'west_circulating'
            ]

        # Color mapping for zone types
        self.zone_colors = {
            'entry': (0, 255, 0),          # Green
            'exit': (0, 0, 255),           # Red (BGR format)
            'queue': (0, 255, 255),        # Yellow
            'circulating': (255, 165, 0),  # Orange
            'intersection': (255, 0, 255)  # Magenta
        }

        self.current_zone_idx = 0

        # Get first frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        self.display_frame = self.frame.copy()
        self.window_name = 'Zone Designer'

    def get_zone_color(self, zone_name):
        """Get color based on zone type"""
        if 'entry' in zone_name:
            return self.zone_colors['entry']
        elif 'exit' in zone_name:
            return self.zone_colors['exit']
        elif 'queue' in zone_name:
            return self.zone_colors['queue']
        elif 'circulating' in zone_name:
            return self.zone_colors['circulating']
        elif 'intersection' in zone_name:
            return self.zone_colors['intersection']
        return (128, 128, 128)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zone drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current zone
            self.current_points.append([x, y])
            print(f"  Added point ({x}, {y}) - Total points: {len(self.current_points)}")
            self.draw_current_state()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish current zone
            if len(self.current_points) >= 3:
                self.zones[self.current_zone_name] = {
                    'points': self.current_points.copy(),
                    'type': self.current_zone_name.split('_')[-1],
                    'direction': self.current_zone_name.split('_')[0]
                }
                print(f"✓ Zone '{self.current_zone_name}' saved with {len(self.current_points)} points")
                print()

                # Check if all zones are complete
                if len(self.zones) == len(self.zone_types):
                    print("=" * 70)
                    print("✓ All zones completed!")
                    print("=" * 70)
                    self.save_zones()
                    print("\nReturning to setup script...")
                    return  # Exit automatically

                # Move to next zone
                self.current_points = []
                self.current_zone_idx = (self.current_zone_idx + 1) % len(self.zone_types)
                self.current_zone_name = self.zone_types[self.current_zone_idx]
                print(f"→ Now drawing: {self.current_zone_name}")
                self.draw_current_state()
            else:
                print("⚠ Need at least 3 points to finish zone")

    def draw_current_state(self):
        """Redraw the display with all zones and current progress"""
        self.display_frame = self.frame.copy()

        # Draw completed zones
        for zone_name, zone_data in self.zones.items():
            points = np.array(zone_data['points'], dtype=np.int32)
            color = self.get_zone_color(zone_name)

            # Draw semi-transparent fill
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)

            # Draw border
            cv2.polylines(self.display_frame, [points], True, color, 2)

            # Draw zone label
            center = points.mean(axis=0).astype(int)
            cv2.putText(self.display_frame, zone_name, tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw current zone in progress
        if len(self.current_points) > 0:
            points = np.array(self.current_points, dtype=np.int32)
            color = self.get_zone_color(self.current_zone_name)

            # Draw points
            for point in self.current_points:
                cv2.circle(self.display_frame, tuple(point), 5, color, -1)

            # Draw lines between points
            if len(self.current_points) > 1:
                cv2.polylines(self.display_frame, [points], False, color, 2)

            # Draw line from last point to first (preview closure)
            if len(self.current_points) >= 3:
                cv2.line(self.display_frame, tuple(self.current_points[-1]),
                        tuple(self.current_points[0]), color, 1)

        # Draw instructions panel
        self._draw_instructions()

        cv2.imshow(self.window_name, self.display_frame)

    def _draw_instructions(self):
        """Draw instruction panel on screen"""
        instructions = [
            f"Zone: {self.current_zone_name} ({self.current_zone_idx + 1}/{len(self.zone_types)})",
            f"Points: {len(self.current_points)}",
            f"Saved: {len(self.zones)}/{len(self.zone_types)}",
            "",
            "Left Click: Add point",
            "Right Click: Finish zone",
            "",
            "S: Save config",
            "L: Load config",
            "C: Clear current",
            "D: Delete last zone",
            "N: Skip to next",
            "Q: Quit"
        ]

        # Draw semi-transparent background
        panel_width = 280
        panel_height = len(instructions) * 25 + 20
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_frame, 0.3, 0, self.display_frame)

        # Draw instructions
        y_offset = 25
        for i, instruction in enumerate(instructions):
            color = (255, 255, 0) if i < 3 else (255, 255, 255)
            cv2.putText(self.display_frame, instruction, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def save_zones(self, output_path=None):
        """Save zones to JSON file"""
        if output_path is None:
            output_path = self.config_path or Path(self.video_path).with_suffix('.zones.json')

        # Convert to format compatible with vehicle_counting.py
        config = {
            "zones": {},
            "zone_colors": {},
            "zone_descriptions": {}
        }

        for zone_name, zone_data in self.zones.items():
            config["zones"][zone_name] = zone_data['points']
            color = self.get_zone_color(zone_name)
            # Convert BGR to hex
            config["zone_colors"][zone_name] = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
            config["zone_descriptions"][zone_name] = f"{zone_data['direction']} {zone_data['type']} zone"

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Configuration saved to {output_path}")
        print(f"  Total zones: {len(self.zones)}")

    def load_zones(self, input_path=None):
        """Load zones from JSON file"""
        if input_path is None:
            input_path = self.config_path or Path(self.video_path).with_suffix('.zones.json')

        try:
            with open(input_path, 'r') as f:
                config = json.load(f)

            # Convert from config format to internal format
            if "zones" in config:
                self.zones = {}
                for zone_name, points in config["zones"].items():
                    # Determine type and direction from zone name
                    parts = zone_name.split('_')
                    direction = parts[0] if len(parts) > 1 else 'unknown'
                    zone_type = parts[-1] if len(parts) > 1 else 'unknown'

                    self.zones[zone_name] = {
                        'points': points,
                        'type': zone_type,
                        'direction': direction
                    }

                print(f"✓ Configuration loaded from {input_path}")
                print(f"  Loaded {len(self.zones)} zones")
                self.draw_current_state()
            else:
                print(f"✗ Invalid configuration format in {input_path}")

        except FileNotFoundError:
            print(f"✗ File {input_path} not found")
        except json.JSONDecodeError:
            print(f"✗ Invalid JSON in {input_path}")

    def run(self):
        """Run the interactive zone designer"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.current_zone_name = self.zone_types[0]

        print("=" * 70)
        print("Zone Designer - Roundabout Traffic Analysis")
        print("=" * 70)
        print(f"Video: {self.video_path}")
        if self.direction:
            print(f"Direction: {self.direction.upper()}")
        print(f"Total zones to define: {len(self.zone_types)}")
        print()
        print("Zone Types:")
        print("  Entry (Green): Vehicles entering the approach")
        print("  Exit (Red): Vehicles exiting the roundabout")
        print("  Queue (Yellow): Queue/waiting area before entry")
        print("  Circulating (Orange): Vehicles circulating in the roundabout")
        print()
        if self.direction:
            print(f"Zones to define for {self.direction.upper()}:")
            for i, zone in enumerate(self.zone_types, 1):
                print(f"  {i}. {zone}")
        print()
        print(f"→ Starting with: {self.current_zone_name}")
        print()

        self.draw_current_state()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break

            elif key == ord('s'):
                self.save_zones()

            elif key == ord('l'):
                self.load_zones()

            elif key == ord('c'):
                # Clear current zone in progress
                if self.current_points:
                    print(f"✓ Cleared {len(self.current_points)} points from current zone")
                    self.current_points = []
                    self.draw_current_state()

            elif key == ord('d'):
                # Delete last saved zone
                if self.zones:
                    last_zone = list(self.zones.keys())[-1]
                    del self.zones[last_zone]
                    print(f"✓ Deleted zone '{last_zone}'")
                    self.draw_current_state()

            elif key == ord('n'):
                # Next zone type (skip current)
                if self.current_points:
                    print("⚠ Clear current points first (press 'c')")
                else:
                    self.current_zone_idx = (self.current_zone_idx + 1) % len(self.zone_types)
                    self.current_zone_name = self.zone_types[self.current_zone_idx]
                    print(f"→ Skipped to zone: {self.current_zone_name}")
                    self.draw_current_state()

        self.cap.release()
        cv2.destroyAllWindows()

        # Offer to save on exit (only if not all zones complete)
        if self.zones and len(self.zones) > 0 and len(self.zones) < len(self.zone_types):
            print()
            print(f"You have {len(self.zones)}/{len(self.zone_types)} zones defined.")
            save = input("Save before exiting? (y/n): ").lower().strip()
            if save == 'y':
                self.save_zones()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive zone designer for roundabout traffic analysis"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to save/load configuration JSON"
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["north", "south", "east", "west"],
        help="Specific direction to define zones for (north, south, east, or west)"
    )

    args = parser.parse_args()

    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    designer = ZoneDesigner(args.video, args.config, args.direction)
    designer.run()


if __name__ == "__main__":
    main()
