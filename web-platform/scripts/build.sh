#!/bin/bash
set -e

# Independent Research - Build Script
# Builds both backend and frontend for distribution

echo "🏗️  Building Independent Research Edge AI Platform..."

# Ensure we're in the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "📍 Working from repository root: $REPO_ROOT"

# Create build directory
mkdir -p dist
rm -rf dist/*

echo "📦 Building Frontend..."
npm run build

echo "🐍 Building Backend..."
# Create Python wheel for distribution (optional)
pip install build 2>/dev/null || echo "Build package not available, skipping wheel creation"
python -m build --wheel 2>/dev/null || echo "Wheel build failed, continuing with source distribution"

echo "📁 Copying distribution files..."

# Copy frontend build (Vite builds to dist/public by default)
if [ -d "dist/public" ]; then
    echo "Copying frontend from dist/public/"
    cp -r dist/public/* dist/
    rm -rf dist/public
elif [ -d "client/dist" ]; then
    echo "Copying frontend from client/dist/"
    cp -r client/dist/* dist/
else
    echo "⚠️  Frontend build not found. Ensure 'npm run build' completed successfully."
fi

# Copy built server file if it exists
if [ -f "dist/index.js" ]; then
    echo "Including built server: dist/index.js"
else
    echo "⚠️  Built server file not found at dist/index.js"
fi

# Copy backend files for distribution  
mkdir -p dist/app
cp -r app/ dist/
cp pyproject.toml start_fastapi.py dist/
cp -r scripts/ dist/

# Copy server files
mkdir -p dist/server
cp -r server/ dist/server/
cp package.json package-lock.json dist/

# Copy configuration and documentation
cp README.md dist/ 2>/dev/null || echo "README.md not found, skipping..."
cp whitepaper.md whitepaper.html dist/ 2>/dev/null || echo "Whitepaper files not found, skipping..."
cp scripts/install.sh dist/ 2>/dev/null || echo "install.sh not found, skipping..."

echo "✅ Build complete! Distribution files are in ./dist/"
echo "📊 Build summary:"
echo "   Frontend: ./dist/index.html (and assets)"
echo "   Backend: ./dist/app/ and ./dist/start_fastapi.py"
echo "   Server: ./dist/server/"
echo "   Configuration: ./dist/pyproject.toml"
echo "   Documentation: ./dist/whitepaper.md and ./dist/whitepaper.html"
echo "   Installer: ./dist/install.sh"