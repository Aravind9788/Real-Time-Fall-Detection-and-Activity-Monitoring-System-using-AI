"""
WebSocket routes for real-time camera feed and alerts
"""

import cv2
import time
import base64
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websockets"])

# Global state (will be injected from main.py)
last_frame = None
frame_queue = None
frame_processor = None
alert_queue = None
connected_alert_clients = set()


def init_websocket_state(
    frame,
    queue,
    processor,
    alerts_queue,
    clients_set
):
    """Initialize global state references from main.py"""
    global last_frame, frame_queue, frame_processor, alert_queue, connected_alert_clients
    last_frame = frame
    frame_queue = queue
    frame_processor = processor
    alert_queue = alerts_queue
    connected_alert_clients = clients_set


@router.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    """
    WebSocket endpoint for receiving camera frames from frontend.
    
    This endpoint runs in the WebSocket receive loop and only handles frame ingestion.
    All processing is delegated to the worker thread via the frame queue.
    """
    await websocket.accept()
    print("✓ Camera WebSocket client connected")
    
    try:
        frame_number = 0
        while True:
            # Receive frame data from client
            data = await websocket.receive_json()
            frame_number += 1
            
            try:
                # Decode base64 image
                img_data_base64 = data.get('frame', '')
                if ',' in img_data_base64:
                    img_data_base64 = img_data_base64.split(',')[1]
                
                img_data = base64.b64decode(img_data_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None and frame.size > 0:
                    # Push frame to queue (non-blocking, drops old frames if full)
                    frame_queue.put({
                        'frame': frame,
                        'timestamp': time.time(),
                        'frame_number': frame_number
                    })
                    
                    # Optionally send acknowledgment (keep it minimal to avoid blocking)
                    # await websocket.send_json({"status": "received"})
                    
            except Exception as e:
                print(f"Error decoding frame: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("✗ Camera WebSocket client disconnected")
    except Exception as e:
        print(f"Camera WebSocket error: {e}")
        import traceback
        traceback.print_exc()


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for broadcasting alerts and processed frames to frontend.
    
    This endpoint sends:
    - Fall detection alerts (MINOR/MAJOR/RECOVERED)
    - Active person tracking data
    - Optional processed frames with bounding boxes
    """
    await websocket.accept()
    connected_alert_clients.add(websocket)
    print(f"✓ Alert WebSocket client connected (total: {len(connected_alert_clients)})")
    
    try:
        while True:
            # Get latest processing result
            result = frame_processor.get_latest_result() if frame_processor else None
            
            if result and 'error' not in result:
                # Prepare data to send
                send_data = {
                    "type": "update",
                    "people": result.get('people', []),
                    "frame_count": result.get('frame_count', 0),
                    "detected_count": result.get('detected_count', 0)
                }
                
                # Optionally include processed frame (comment out if you want alerts-only)
                if 'processed_frame' in result:
                    _, buffer = cv2.imencode('.jpg', result['processed_frame'], [cv2.IMWRITE_JPEG_QUALITY, 70])
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    send_data['processed_frame'] = jpg_as_text
                
                # Send update
                await websocket.send_json(send_data)
            else:
                # Send blank/waiting frame if no result yet
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Waiting for camera...", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(blank_frame, "Starting frame processor...", (120, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                _, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    "type": "update",
                    "processed_frame": jpg_as_text,
                    "detected_count": 0,
                    "frame_count": 0,
                    "people": []
                })
            
            # Check for alerts in the queue
            try:
                # Non-blocking check for alerts
                while not alert_queue.empty():
                    alert_data = await asyncio.wait_for(alert_queue.get(), timeout=0.01)
                    await websocket.send_json({
                        "type": "alert",
                        **alert_data
                    })
            except asyncio.TimeoutError:
                pass
            
            # Small delay to avoid overwhelming the client (~20 FPS)
            await asyncio.sleep(0.05)
            
    except WebSocketDisconnect:
        connected_alert_clients.discard(websocket)
        print(f"✗ Alert WebSocket client disconnected (remaining: {len(connected_alert_clients)})")
    except Exception as e:
        connected_alert_clients.discard(websocket)
        print(f"Alert WebSocket error: {e}")
        import traceback
        traceback.print_exc()


@router.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    await websocket.accept()
    print("✓ WebSocket client connected")
    try:
        while True:
            # Get current frame from global last_frame
            if last_frame is not None:
                # Encode frame as JPEG for efficient transmission
                _, buffer = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                # Convert to base64 for WebSocket transmission
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                # Send to client
                await websocket.send_json({
                    "type": "frame",
                    "data": jpg_as_text
                })
            await asyncio.sleep(0.033)  # ~30 FPS
    except WebSocketDisconnect:
        print("✗ WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
