"""
View routes for HTML templates
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["views"])

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Enhanced modern dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})
