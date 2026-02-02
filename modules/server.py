"""
Server management module.

Handles FastAPI server startup and configuration.
"""

import uvicorn


def run_server(app, host="127.0.0.1", port=8000, log_level="info"):
    """
    Run FastAPI server with uvicorn.
    
    Args:
        app: FastAPI application instance
        host: Server host address
        port: Server port
        log_level: Logging level
    """
    try:
        print("FastAPI: Starting server...")
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    except Exception as e:
        print(f"FastAPI Error: {e}")
        import traceback
        traceback.print_exc()
