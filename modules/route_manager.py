"""
Route management module.

Handles initialization and registration of all API routes.
"""

from collections import defaultdict


def init_routes(app, state_manager, last_frame, all_tracked_people,
                model, rename_person_func, register_face, manual_id_map,
                data_lock, DB_PATH, format_duration, get_fall_history,
                frame_queue, frame_processor, alert_queue, connected_alert_clients,
                fall, active_alerts):
    """
    Initialize all route modules with global state and dependencies.
    
    Args:
        app: FastAPI application instance
        state_manager: ActivityStateManager instance
        last_frame: Last captured frame
        all_tracked_people: Set of all tracked people
        model: YOLO model instance
        rename_person_func: Function to rename a person
        register_face: Face registration function
        manual_id_map: Dictionary mapping IDs to names
        data_lock: Threading lock
        DB_PATH: Database path
        format_duration: Function to format duration
        get_fall_history: Function to get fall history
        frame_queue: Frame processing queue
        frame_processor: Frame processor instance
        alert_queue: Alert queue for WebSocket
        connected_alert_clients: Set of WebSocket clients
        fall: Fall flag
        active_alerts: List of active alerts
    """
    # Import route modules
    from routes import alerts, registration, api, websockets
    from routes import (
        alerts_router,
        registration_router,
        api_router,
        websockets_router,
        views_router
    )
    
    # Initialize alerts routes
    alerts.fall = fall
    alerts.active_alerts = active_alerts
    
    # Initialize registration routes
    registration.last_frame = last_frame
    registration.person_state = state_manager.person_state if state_manager else {}
    registration.all_tracked_people = all_tracked_people
    registration.model = model
    registration.rename_person_func = rename_person_func
    registration.register_face_func = register_face
    
    # Initialize API routes
    api.person_state = state_manager.person_state if state_manager else {}
    api.all_tracked_people = all_tracked_people
    api.manual_id_map = manual_id_map
    api.walking_time = state_manager.walking_time if state_manager else defaultdict(float)
    api.sitting_time = state_manager.sitting_time if state_manager else defaultdict(float)
    api.sleeping_time = state_manager.sleeping_time if state_manager else defaultdict(float)
    api.standing_time = state_manager.standing_time if state_manager else defaultdict(float)
    api.active_alerts = active_alerts
    api.data_lock = data_lock
    api.DB_PATH = DB_PATH
    api.format_duration_func = format_duration
    api.get_fall_history_func = get_fall_history
    
    # Initialize WebSocket routes
    websockets.last_frame = last_frame
    websockets.frame_queue = frame_queue
    websockets.frame_processor = frame_processor
    websockets.alert_queue = alert_queue
    websockets.connected_alert_clients = connected_alert_clients
    
    # Include all routers
    app.include_router(alerts_router)
    app.include_router(registration_router)
    app.include_router(api_router)
    app.include_router(websockets_router)
    app.include_router(views_router)
