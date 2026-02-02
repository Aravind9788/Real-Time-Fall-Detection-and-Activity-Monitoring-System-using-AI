"""
Routes package for Elderly Activity Monitor
"""

from fastapi import APIRouter

# Import all route modules
from .alerts import router as alerts_router
from .registration import router as registration_router
from .api import router as api_router
from .websockets import router as websockets_router
from .views import router as views_router

# Collect all routers
__all__ = [
    "alerts_router",
    "registration_router",
    "api_router",
    "websockets_router",
    "views_router"
]
