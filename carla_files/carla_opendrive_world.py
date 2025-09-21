#!/usr/bin/env python3
"""
CARLA Simple World Builder  
- Uses built-in CARLA map (Town01)
- Finds straight road section
- Places camera roadside at angle
- Spawns 2 construction cones at 5m and 10m
- Spawns vehicle approaching along lane
- MODIFIED: Camera view cropped to 550x500 (removes borders: left 50px, right 200px, top 100px)
"""

import carla
import numpy as np
import queue
import math
import time

class CarlaSimpleWorld:
    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.construction_cones = []
        self.test_vehicle = None
        self.image_queue = queue.Queue()
        
        # Camera and cone positioning
        self.camera_location = None
        self.road_reference_point = None
        
    def setup_world(self, host='localhost', port=2000):
        """Setup CARLA world with built-in map"""
        print("Setting up CARLA world...")
        
        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        try:
            # Load a simple built-in map
            self.world = self.client.load_world('Town01')
            print("Loaded Town01 map")
            
            # Wait for world to stabilize
            time.sleep(2)
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
        
        # Clear existing actors
        self._clear_all_actors()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Setup world components
        self._setup_roadside_camera()
        self._place_construction_cones()
        self._spawn_approaching_vehicle()
        
        print("CARLA world setup complete!")
        return True
    
    def _clear_all_actors(self):
        """Clear all existing actors"""
        actors_to_destroy = []
        for actor in self.world.get_actors():
            if ('vehicle' in actor.type_id.lower() or 
                'cone' in actor.type_id.lower() or
                'construction' in actor.type_id.lower()):
                actors_to_destroy.append(actor)
        
        for actor in actors_to_destroy:
            try:
                actor.destroy()
            except:
                pass
    
    def _setup_roadside_camera(self):
        """Setup camera roadside on a straight section of road"""
        print("Setting up roadside camera...")
        
        # Get spawn points and find a good straight section
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Use a spawn point that's likely on a straight road
        selected_spawn = None
        for i, spawn_point in enumerate(spawn_points):
            # Skip early spawn points (often at intersections)
            if 10 <= i <= 20:  # Pick from middle range
                selected_spawn = spawn_point
                break
        
        if not selected_spawn:
            selected_spawn = spawn_points[15]  # Fallback
        
        print(f"Using spawn point: {selected_spawn.location}")
        
        # Get road direction vectors
        forward_vector = selected_spawn.rotation.get_forward_vector()
        right_vector = selected_spawn.rotation.get_right_vector()
        
        # Position road reference point ahead on the road
        road_position = carla.Location(
            x=selected_spawn.location.x + forward_vector.x * 20.0,
            y=selected_spawn.location.y + forward_vector.y * 20.0,
            z=selected_spawn.location.z
        )
        
        # Camera positioned roadside, looking at angle toward road
        camera_location = carla.Location(
            x=road_position.x + right_vector.x * 4.5,  # 4.5m right of road
            y=road_position.y + right_vector.y * 4.5,
            z=road_position.z + 1.5  # 1.5m high
        )
        
        # Camera rotation - look toward road at angle
        camera_rotation = carla.Rotation(
            pitch=-8,   # Downward angle to see road surface
            yaw=selected_spawn.rotation.yaw - 25,  # 25° toward road
            roll=0
        )
        
        camera_transform = carla.Transform(camera_location, camera_rotation)
        
        # Create RGB camera - keep at 800x600 for processing, will crop to 700x600
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        self.camera = self.world.spawn_actor(camera_bp, camera_transform)
        self.camera.listen(self._on_camera_data)
        
        # Store reference points
        self.camera_location = camera_location
        self.road_reference_point = carla.Transform(
            road_position,
            selected_spawn.rotation
        )
        
        print(f"Camera positioned at: {camera_location}")
        print(f"Camera rotation: {camera_rotation}")
        print("Camera will output 550x500 images (cropped from 800x600)")
        return True
    
    def _place_construction_cones(self):
        """Place construction cones on roadside shoulder at 5m and 10m distances"""
        print("Placing construction cones on roadside shoulder...")
        
        cone_bp = self.blueprint_library.find('static.prop.constructioncone')
        if not cone_bp:
            cone_bp = self.blueprint_library.find('static.prop.trafficcone01')
        
        if not cone_bp:
            print("No suitable cone blueprints found")
            return False
        
        # Get road direction (forward along the road)
        forward_vector = self.road_reference_point.rotation.get_forward_vector()
        right_vector = self.road_reference_point.rotation.get_right_vector()
        
        # Cone distances along the road - EXACTLY 5m and 10m for calibration
        ref_distances = [5.0, 10.0]
        
        for i, distance in enumerate(ref_distances):
            # Place cones on roadside shoulder (closer to road edge)
            cone_location = carla.Location(
                x=self.road_reference_point.location.x + forward_vector.x * distance + right_vector.x * 3.0,
                y=self.road_reference_point.location.y + forward_vector.y * distance + right_vector.y * 3.0,
                z=self.road_reference_point.location.z + 0.1
            )
            
            cone_transform = carla.Transform(
                cone_location,
                self.road_reference_point.rotation
            )
            
            try:
                cone = self.world.spawn_actor(cone_bp, cone_transform)
                if hasattr(cone, 'set_simulate_physics'):
                    cone.set_simulate_physics(False)
                
                self.construction_cones.append(cone)
                print(f"Construction cone {i+1} placed at {distance}m along road (shoulder)")
                print(f"  Location: {cone_location}")
            except Exception as e:
                print(f"Could not place cone {i+1}: {e}")
        
        return len(self.construction_cones) > 0
    
    def _spawn_approaching_vehicle(self):
        """Spawn vehicle that approaches camera along the road"""
        print("Spawning approaching vehicle...")
        
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        if not vehicle_bp:
            # Try alternative vehicle blueprints
            vehicle_blueprints = [
                'vehicle.audi.a2',
                'vehicle.bmw.grandtourer',
                'vehicle.chevrolet.impala',
                'vehicle.ford.mustang',
                'vehicle.toyota.prius'
            ]
            for bp_name in vehicle_blueprints:
                vehicle_bp = self.blueprint_library.find(bp_name)
                if vehicle_bp:
                    break
        
        if not vehicle_bp:
            print("No suitable vehicle blueprint found")
            return False
        
        # Get road direction
        forward_vector = self.road_reference_point.rotation.get_forward_vector()
        
        # Spawn vehicle ahead on the road (start at 35m distance)
        vehicle_location = carla.Location(
            x=self.road_reference_point.location.x + forward_vector.x * 35.0,
            y=self.road_reference_point.location.y + forward_vector.y * 35.0,
            z=self.road_reference_point.location.z + 0.5
        )
        
        # Vehicle faces toward camera (opposite of road direction)
        vehicle_rotation = carla.Rotation(
            pitch=0,
            yaw=self.road_reference_point.rotation.yaw + 180,
            roll=0
        )
        
        vehicle_transform = carla.Transform(vehicle_location, vehicle_rotation)
        
        try:
            self.test_vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)
            print(f"Vehicle spawned - will approach camera")
            print(f"  Location: {vehicle_location}")
            print(f"  Model: {vehicle_bp.id}")
            return True
        except Exception as e:
            print(f"Could not spawn vehicle: {e}")
            return False
    
    def _control_vehicle(self):
        """Control vehicle to drive straight toward camera"""
        if self.test_vehicle:
            control = carla.VehicleControl()
            control.throttle = 0.3  # Moderate speed
            control.steer = 0.0     # Drive straight
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            
            self.test_vehicle.apply_control(control)
    
    def _on_camera_data(self, image):
        """Camera data callback - crops image by removing borders"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]  # BGRA to RGB
        
        # Crop image: remove left 50px, right 200px, top 100px
        # Original: 800x600 -> Final: 550x500
        # Keep columns 50 to 599 (remove left 50, right 200)
        # Keep rows 100 to 599 (remove top 100)
        cropped_array = array[100:600, 50:600, :]
        
        if not self.image_queue.full():
            self.image_queue.put(cropped_array)
    
    def get_frame(self):
        """Get next frame from camera (now 550x500)"""
        self.world.tick()
        self._control_vehicle()
        
        try:
            return self.image_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def cleanup(self):
        """Clean up all CARLA actors"""
        print("Cleaning up CARLA world...")
        
        if self.camera:
            self.camera.destroy()
        for cone in self.construction_cones:
            cone.destroy()
        if self.test_vehicle:
            self.test_vehicle.destroy()
        
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        print("Cleanup complete")

def main():
    """
    Standalone CARLA world for testing
    Run this first, then run detection_system_carla.py in another terminal
    """
    print("=== CARLA World Setup ===")
    print("This creates the CARLA simulation environment")
    print("Run detection_system_carla.py in another terminal to start detection")
    
    # Create and setup world
    world = CarlaSimpleWorld()
    
    if not world.setup_world():
        print("Failed to setup world")
        return
    
    print("\nCARLA World Ready!")
    print("- Built-in Town01 map with proper roads")
    print("- Camera positioned roadside (550x500 output)")
    print("- Construction cones at 5m and 10m")
    print("- Vehicle approaching along road")
    print("- Ready for detection system integration")
    print("\nNow run: python detection_system_carla.py")
    print("Press Ctrl+C to stop this world")
    
    try:
        # Keep world running
        frame_count = 0
        while True:
            frame = world.get_frame()
            if frame is not None:
                frame_count += 1
                if frame_count % 100 == 0:  # Print status every 100 frames
                    print(f"World running... Frame {frame_count} (550x500)")
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nStopping world...")
    finally:
        world.cleanup()

if __name__ == "__main__":
    main()