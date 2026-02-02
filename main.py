"""
Elderly Activity Monitor - Main Entry Point

This is the main entry point for the elderly monitoring system with fall detection,
activity tracking, and real-time web dashboard.
"""

import time
import sys
import threading
import atexit

# Import custom modules
from modules.reid import reid_manager
from modules.face_rec import register_face, data_lock
from modules.detection import model, classify_activity, format_duration
from modules.database import (
    load_manual_id_map, save_manual_id_map, load_stats_from_db, 
    send_fall_alert, DB_PATH, manual_id_map, get_fall_history,
    log_activity_to_db
)
from modules.frame_processor import FrameQueue, FrameProcessor
from modules.tracker_logic import PersonTracker
from modules.state_manager import ActivityStateManager
from modules.app_config import create_app
from modules.helpers import rename_person
from modules.route_manager import init_routes
from modules.server import run_server

# Import global state from app_config
import modules.app_config as app_config


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Create FastAPI app
    print("Creating FastAPI application...")
    app = create_app()
    
    # Initialize new module instances
    print("Initializing tracking and state management modules...")
    person_tracker = PersonTracker(reid_manager)
    state_manager = ActivityStateManager(send_fall_alert)
    
    # Store in global app_config for access by other modules
    app_config.person_tracker = person_tracker
    app_config.state_manager = state_manager
    
    # Load stats from database into state_manager
    load_stats_from_db(state_manager.walking_time, state_manager.sitting_time, 
                       state_manager.sleeping_time, app_config.all_tracked_people)
    
    # Initialize frame queue and processor
    print("Initializing frame processing system...")
    app_config.frame_queue = FrameQueue(maxsize=2)
    
    # Alert callback function to push alerts to WebSocket queue
    def handle_alert(alert_data):
        """Push alert to queue for WebSocket broadcasting"""
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(
                app_config.alert_queue.put(alert_data), 
                asyncio.get_event_loop()
            )
        except Exception as e:
            print(f"Error queuing alert: {e}")
    
    # Create frame processor
    app_config.frame_processor = FrameProcessor(
        frame_queue=app_config.frame_queue,
        alert_callback=handle_alert
    )
    
    # Create wrapper for rename_person that includes all dependencies
    def rename_person_wrapper(old_id, new_name):
        """Wrapper for rename_person with all dependencies"""
        return rename_person(
            old_id, new_name,
            state_manager=state_manager,
            all_tracked_people=app_config.all_tracked_people,
            manual_id_map=manual_id_map,
            save_manual_id_map_func=save_manual_id_map,
            data_lock=data_lock
        )
    
    # Inject dependencies into frame processor (now using full objects)
    app_config.frame_processor.inject_dependencies(
        model=model,
        person_tracker=person_tracker,
        state_manager=state_manager,
        app_config=app_config,
        detected_ids=app_config.detected_ids,
        last_detection=app_config.last_detection,
        classify_activity=classify_activity,
        rename_person=rename_person_wrapper,
        reid_manager=reid_manager,
        known_face_encodings=__import__('modules.face_rec', fromlist=['known_face_encodings']).known_face_encodings,
        known_face_names=__import__('modules.face_rec', fromlist=['known_face_names']).known_face_names
    )
    
    # Start frame processor worker thread
    app_config.frame_processor.start()
    
    # Register shutdown handler
    def cleanup():
        """Graceful shutdown of frame processor"""
        print("\nShutting down frame processor...")
        if app_config.frame_processor:
            app_config.frame_processor.stop()
    
    atexit.register(cleanup)
    
    # Start DB sync thread (now using state_manager's time dictionaries)
    db_sync_thread = threading.Thread(
        target=log_activity_to_db, 
        args=(app_config.all_tracked_people, state_manager.walking_time, state_manager.sitting_time, 
              state_manager.sleeping_time, data_lock, reid_manager, save_manual_id_map),
        daemon=True
    )
    db_sync_thread.start()
    
    # Initialize all routes with global state
    print("Initializing routes...")
    init_routes(
        app=app,
        state_manager=state_manager,
        last_frame=app_config.last_frame,
        all_tracked_people=app_config.all_tracked_people,
        model=model,
        rename_person_func=rename_person_wrapper,
        register_face=register_face,
        manual_id_map=manual_id_map,
        data_lock=data_lock,
        DB_PATH=DB_PATH,
        format_duration=format_duration,
        get_fall_history=get_fall_history,
        frame_queue=app_config.frame_queue,
        frame_processor=app_config.frame_processor,
        alert_queue=app_config.alert_queue,
        connected_alert_clients=app_config.connected_alert_clients,
        fall=app_config.fall,
        active_alerts=app_config.active_alerts
    )
    
    # Start FastAPI server thread
    print("Starting FastAPI server thread...")
    server_thread = threading.Thread(
        target=run_server, 
        args=(app,),
        daemon=True
    )
    server_thread.start()
    
    time.sleep(1)  # Give FastAPI time to start
    print("✓ FastAPI server running at http://127.0.0.1:8000")
    print("  Access the dashboard at: http://127.0.0.1:8000/")
    print("  WebSocket camera endpoint: ws://127.0.0.1:8000/ws/camera")
    print("  WebSocket alerts endpoint: ws://127.0.0.1:8000/ws/alerts")
    print("\n✓ Real-time frame processing system initialized")
    print(f"  Frame queue size: 5 frames (~165ms latency at 30 FPS)")
    print("  Worker thread: Running")
    print("  Processing mode: Sequential (one frame at a time for accuracy)")
    
    # Keep the server running
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nProgram exited.")
        cleanup()
        sys.exit(0)
