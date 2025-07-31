# Makefile for AWS Bedrock Plugin for Genkit Go

.PHONY: help build test lint clean examples release-test release-snapshot

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build targets
build: ## Build all examples
	@echo "Building examples..."
	@cd examples/basic && go build -v .
	@cd examples/advanced_schemas && go build -v .
	@cd examples/embeddings && go build -v .
	@cd examples/image_generation && go build -v .
	@cd examples/multimodal && go build -v .
	@cd examples/streaming && go build -v .
	@cd examples/tool_calling && go build -v .
	@echo "✅ All examples built successfully"

# Test targets
test: ## Run all tests
	@echo "Running tests..."
	@go test -v ./...

test-race: ## Run tests with race detection
	@echo "Running tests with race detection..."
	@go test -race -v ./...

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	@go test -race -coverprofile=coverage.out -covermode=atomic ./...
	@go tool cover -html=coverage.out -o coverage.html
	@echo "✅ Coverage report generated: coverage.html"

# Linting
lint: ## Run linter
	@echo "Running linter..."
	@golangci-lint run --timeout=5m

lint-fix: ## Run linter with auto-fix
	@echo "Running linter with auto-fix..."
	@golangci-lint run --fix --timeout=5m

# Dependencies
deps: ## Download and verify dependencies
	@echo "Downloading dependencies..."
	@go mod download
	@go mod verify

tidy: ## Tidy up dependencies
	@echo "Tidying up dependencies..."
	@go mod tidy

# Examples
examples: build ## Build and run basic example (requires AWS credentials)
	@echo "Running basic example..."
	@cd examples/basic && ./basic

examples-all: build ## Build all examples (requires AWS credentials)
	@echo "Note: These examples require valid AWS credentials and access to AWS Bedrock"
	@echo "Running basic example..."
	@cd examples/basic && timeout 30s ./basic || true
	@echo "Running streaming example..."
	@cd examples/streaming && timeout 30s ./streaming || true
	@echo "Running tool calling example..."
	@cd examples/tool_calling && timeout 30s ./tool_calling || true

# Release targets
release-test: ## Test the release process locally
	@echo "Testing release process..."
	@goreleaser check
	@goreleaser build --snapshot --clean
	@echo "✅ Release test completed"

release-snapshot: ## Create a snapshot release (no tags)
	@echo "Creating snapshot release..."
	@goreleaser release --snapshot --clean
	@echo "✅ Snapshot release created in dist/"

# Cleanup
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@rm -rf dist/
	@rm -f coverage.out coverage.html
	@find examples/ -name examples -type f -delete
	@find examples/ -name examples.exe -type f -delete
	@find examples/ -name "*.test" -delete
	@echo "✅ Cleaned build artifacts"

# Development helpers
fmt: ## Format code
	@echo "Formatting code..."
	@go fmt ./...

vet: ## Run go vet
	@echo "Running go vet..."
	@go vet ./...

install-tools: ## Install development tools
	@echo "Installing development tools..."
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@go install github.com/goreleaser/goreleaser@latest
	@echo "✅ Development tools installed"

# Comprehensive checks
check: deps tidy fmt vet lint test ## Run all checks (format, vet, lint, test)
	@echo "✅ All checks passed"

# Git helpers
tag: ## Create a new git tag (usage: make tag VERSION=v1.0.0)
	@if [ -z "$(VERSION)" ]; then echo "Usage: make tag VERSION=v1.0.0"; exit 1; fi
	@echo "Creating tag $(VERSION)..."
	@git tag -a $(VERSION) -m "Release $(VERSION)"
	@echo "✅ Tag $(VERSION) created. Push with: git push origin $(VERSION)"

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	@go doc -all . > docs.txt
	@echo "✅ Documentation generated in docs.txt"

# Release checklist
release-checklist: ## Show release checklist
	@echo "Release Checklist:"
	@echo "1. ✅ Run 'make check' to ensure all tests pass"
	@echo "2. ✅ Update README.md if needed"
	@echo "3. ✅ Update CHANGELOG.md if maintained"
	@echo "4. ✅ Run 'make release-test' to test locally"
	@echo "5. ✅ Create and push a new tag: make tag VERSION=vX.Y.Z && git push origin vX.Y.Z"
	@echo "6. ✅ Monitor the GitHub Actions release workflow"
	@echo "7. ✅ Verify the release on GitHub"
