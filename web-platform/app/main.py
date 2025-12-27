"""
FastAPI application factory and main entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.core.config import get_settings
from app.core.database import create_db_and_tables
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await create_db_and_tables()
    yield
    # Shutdown (cleanup if needed)


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Independent Research Edge AI Platform",
        description="""A comprehensive REST API for managing edge AI devices and enabling 
        large language model (LLM) chat functionality on edge computing hardware.
        
        ## Features
        - **Device Management**: Connect and manage edge devices (Raspberry Pi, NVIDIA Jetson, etc.)
        - **LLM Chat**: Chat with AI models running locally on edge devices
        - **Admin Interface**: Manage devices and backend services
        - **Multimodal Support**: Text and image inputs for AI conversations
        - **Authentication**: Session-based authentication with OIDC support
        
        ## Benefits
        - **Zero Incremental Cost**: No API calls to cloud services
        - **Low Latency**: Local processing for real-time responses
        - **Privacy**: All data stays on your edge devices
        - **Scalability**: Fixed cost regardless of usage volume
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Independent Research",
            "url": "https://independent-research.example.com",
        },
        license_info={
            "name": "MIT",
        }
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5000", "http://localhost:5173"],  # Add frontend origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router with versioning
    app.include_router(api_router, prefix="/api/v1", tags=["API v1"])
    
    # Legacy support for /api endpoints (redirect to v1)
    @app.get("/api/health", tags=["Health"], deprecated=True)
    async def legacy_health_check():
        """Legacy health check endpoint. Use /health instead."""
        return {"status": "healthy", "service": "independent-research-api", "message": "Please use /health endpoint"}
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Check the health status of the API service."""
        return {
            "status": "healthy", 
            "service": "independent-research-api",
            "version": "1.0.0",
            "description": "Edge AI Platform API"
        }
    
    return app


app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )