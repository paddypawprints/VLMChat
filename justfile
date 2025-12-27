# VLMChat Platform Build System
# Uses `just` command runner (https://github.com/casey/just)
# Install: brew install just

set dotenv-load := true

# Default recipe shows available commands
default:
    @just --list

# ============================================================================
# Installation & Setup
# ============================================================================

# Install all dependencies (TypeScript + Python)
install:
    @echo "📦 Installing dependencies..."
    cd web-platform && npm install
    @echo "✅ Dependencies installed"

# Install code generation tools
install-generators:
    @echo "📦 Installing code generators..."
    cd web-platform && npm install -D openapi-typescript @openapitools/openapi-generator-cli
    pip install datamodel-code-generator[http] pydantic
    @echo "✅ Generators installed"

# ============================================================================
# Code Generation
# ============================================================================

# Generate all code from specs (TypeScript + Python)
generate: generate-ts generate-python
    @echo "✅ All code generated"

# Generate TypeScript types from OpenAPI
generate-ts:
    @echo "🔨 Generating TypeScript types from OpenAPI..."
    npx openapi-typescript shared/specs/openapi.yaml \
        --output web-platform/shared/types/api.ts
    @echo "✅ TypeScript types generated"

# Generate Python SDK from JSON schemas
generate-python:
    @echo "🔨 Generating Python SDK from JSON schemas..."
    mkdir -p device-sdk/edge_llm_client/models
    datamodel-codegen \
        --input shared/schemas \
        --input-file-type jsonschema \
        --output device-sdk/edge_llm_client/models \
        --output-model-type pydantic_v2.BaseModel \
        --field-constraints \
        --use-standard-collections \
        --target-python-version 3.10
    @echo "✅ Python SDK generated"

# Generate AsyncAPI documentation
generate-docs:
    @echo "📚 Generating AsyncAPI documentation..."
    npx @asyncapi/generator shared/specs/asyncapi.yaml @asyncapi/html-template \
        --output docs/asyncapi \
        --force-write
    @echo "✅ Documentation generated at docs/asyncapi/index.html"

# ============================================================================
# Development
# ============================================================================

# Start all services with Docker Compose
dev:
    cd web-platform && docker-compose up

# Start services in background
dev-bg:
    cd web-platform && docker-compose up -d

# Stop all services
stop:
    cd web-platform && docker-compose down

# Restart services
restart: stop dev-bg

# View logs from all services
logs service="":
    cd web-platform && docker-compose logs -f {{service}}

# ============================================================================
# Database
# ============================================================================

# Push database schema changes
db-push:
    cd web-platform && npm run db:push

# Open database studio (Drizzle Studio)
db-studio:
    cd web-platform && npx drizzle-kit studio

# ============================================================================
# Building & Type Checking
# ============================================================================

# Type check TypeScript
check:
    cd web-platform && npm run check

# Build web platform for production
build: generate
    cd web-platform && npm run build

# Build vlmchat Python package
build-vlmchat:
    cd vlmchat && pip install -e .

# Build device SDK Python package
build-device-sdk: generate-python
    cd device-sdk && pip install -e .

# ============================================================================
# Testing
# ============================================================================

# Run all tests
test: test-ts test-python

# Run TypeScript tests
test-ts:
    cd web-platform && npm test

# Run Python tests
test-python:
    cd vlmchat && pytest

# ============================================================================
# Validation
# ============================================================================

# Validate OpenAPI spec
validate-openapi:
    @echo "🔍 Validating OpenAPI specification..."
    npx @apidevtools/swagger-cli validate shared/specs/openapi.yaml
    @echo "✅ OpenAPI spec is valid"

# Validate AsyncAPI spec
validate-asyncapi:
    @echo "🔍 Validating AsyncAPI specification..."
    npx @asyncapi/cli validate shared/specs/asyncapi.yaml
    @echo "✅ AsyncAPI spec is valid"

# Validate all specs
validate: validate-openapi validate-asyncapi

# Validate JSON schemas
validate-schemas:
    @echo "🔍 Validating JSON schemas..."
    @for schema in shared/schemas/*/v*/schema.json; do \
        echo "  Checking $$schema..."; \
        python -m json.tool $$schema > /dev/null || exit 1; \
    done
    @echo "✅ All schemas are valid JSON"

# ============================================================================
# Cleaning
# ============================================================================

# Clean generated files
clean:
    @echo "🧹 Cleaning generated files..."
    rm -rf web-platform/dist
    rm -rf web-platform/shared/types
    rm -rf device-sdk/edge_llm_client/models
    rm -rf docs/asyncapi
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    @echo "✅ Clean complete"

# Clean everything including dependencies
clean-all: clean
    @echo "🧹 Cleaning dependencies..."
    rm -rf web-platform/node_modules
    rm -rf vlmchat/.venv
    rm -rf device-sdk/.venv
    @echo "✅ Deep clean complete"

# ============================================================================
# Utilities
# ============================================================================

# Show project structure
tree:
    @echo "📁 Project structure:"
    @find . -maxdepth 2 -type d | grep -E "(shared|vlmchat|web-platform|device-sdk)" | sort

# Show current versions
versions:
    @echo "📌 Tool versions:"
    @echo "Node: $(node --version)"
    @echo "npm: $(npm --version)"
    @echo "Python: $(python3 --version)"
    @echo "Just: $(just --version)"

# Open API documentation in browser
docs-api:
    @echo "🌐 Opening OpenAPI docs..."
    open https://editor.swagger.io/?url=file://$(pwd)/shared/specs/openapi.yaml

# Run format checks
format:
    cd web-platform && npx prettier --write "client/**/*.{ts,tsx}" "server/**/*.ts"
    cd vlmchat && black src/
