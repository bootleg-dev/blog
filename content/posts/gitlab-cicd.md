---
title: "GitLab CI/CD"
description: "How to configure Gitlab CI/CD?"
dateString: "Date: 10 August, 2025"
date: "2025-03-18T18:34:44.165668+0500"
draft: false
tags: ["Beginner", "GitLab", "CI/CD", "Automation"]
weight: 1
cover:
    image: ""
---



# GitLab CI/CD: Complete Setup Guide with Docker

GitLab CI/CD is a powerful tool that automates your software deployment process. In this guide, we'll walk through setting up a complete CI/CD pipeline that builds Docker images and deploys your application automatically.

## What You'll Learn

- How to set up GitLab Runner with Docker
- Creating a CI/CD pipeline configuration
- Building and deploying Docker containers
- Implementing health checks and monitoring

## Prerequisites

- A GitLab account and repository
- Docker installed on your server
- Basic knowledge of Git and Docker

## Step 1: Setting Up GitLab Runner

First, we need to set up a GitLab Runner that can execute our CI/CD jobs.

### Initial Runner Setup

Create a directory for the GitLab Runner configuration:

```bash
mkdir -p $HOME/gitlab-runner/config
```

Start the GitLab Runner container with Docker socket access:

```bash
docker run -d --name gitlab-runner --restart always \
  -v $HOME/gitlab-runner/config:/etc/gitlab-runner \
  -v /var/run/docker.sock:/var/run/docker.sock \
  gitlab/gitlab-runner:latest
```

### Registering the Runner

Now register your runner with GitLab using your project's registration token:

```bash
docker exec -it gitlab-runner gitlab-runner register \
  --non-interactive \
  --url "https://gitlab.com/" \
  --registration-token "YOUR_PROJECT_TOKEN" \
  --executor "docker" \
  --docker-image "docker:24.0.5" \
  --docker-volumes "/var/run/docker.sock:/var/run/docker.sock" \
  --description "docker-runner" \
  --tag-list "docker"
```

**Note**: Replace `YOUR_PROJECT_TOKEN` with your actual project registration token from GitLab Settings > CI/CD > Runners.

## Step 2: Creating the Dockerfile

Our Dockerfile uses a multi-stage build for optimized image size and security:

```dockerfile
# Multi-stage build for Python application
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Key Features of This Dockerfile

- **Multi-stage build**: Separates build dependencies from runtime
- **Security**: Runs as non-root user
- **Optimization**: Removes unnecessary packages and cleans cache
- **Health checks**: Built-in container health monitoring

## Step 3: Creating the CI/CD Pipeline

Create a `.gitlab-ci.yml` file in your project root:

```yaml
stages:
  - docker-build
  - deploy

variables:
  PYTHON_VERSION: "3.11"
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  IMAGE_NAME: python-cicd-app:latest

cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .cache/pip
    - venv/

docker-build:
  stage: docker-build
  image: docker:24.0.5
  variables:
    DOCKER_HOST: unix:///var/run/docker.sock
  before_script:
    - docker info
  script:
    - echo "Building Docker image with socket binding..."
    - docker build -t $IMAGE_NAME .
    - echo "Image built successfully!"
    - docker images | grep python-cicd-app || echo "Image not found in list"
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  tags:
    - docker

deploy:
  stage: deploy
  image: docker:24.0.5
  variables:
    DOCKER_HOST: unix:///var/run/docker.sock
  before_script:
    - apk add --no-cache curl
  script:
    - echo "Deploying Docker container with socket binding..."
    - echo "Stopping existing container..."
    - docker stop python-cicd-app || true
    - docker rm python-cicd-app || true
    - echo "Starting new container..."
    - |
      docker run -d \
        --name python-cicd-app \
        --restart unless-stopped \
        -p 8000:8000 \
        -e ENVIRONMENT=production \
        -e CI_COMMIT_SHA=$CI_COMMIT_SHA \
        -e CI_RUNNER_DESCRIPTION="$CI_RUNNER_DESCRIPTION" \
        $IMAGE_NAME
    - echo "Deployment complete!"
    - sleep 10
    - echo "Container status:"
    - docker ps | grep python-cicd-app
    - echo ""
    - echo "Application URLs:"
    - echo "http://localhost:8000"
    - echo "http://localhost:8000/docs"
    - echo "http://localhost:8000/health"
    - echo ""
    - echo "Testing deployment:"
    - timeout 30 sh -c 'until curl -f http://localhost:8000/health; do echo "Waiting for app..."; sleep 2; done' && echo "✅ App is healthy!" || echo "❌ Health check failed"
  environment:
    name: production
    url: http://localhost:8000
  dependencies:
    - docker-build
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  when: manual
  tags:
    - docker
```

## Understanding the Pipeline

### Pipeline Structure

Our pipeline has two main stages:

1. **docker-build**: Builds the Docker image
2. **deploy**: Deploys the container to production

### Key Components Explained

#### Variables Section
```yaml
variables:
  PYTHON_VERSION: "3.11"
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  IMAGE_NAME: python-cicd-app:latest
```

These are global variables used throughout the pipeline:
- `PYTHON_VERSION`: Specifies Python version
- `PIP_CACHE_DIR`: Directory for pip cache
- `IMAGE_NAME`: Name for our Docker image

#### Caching
```yaml
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .cache/pip
    - venv/
```

Caching speeds up builds by storing pip packages and virtual environments between runs.

#### Docker Build Job
```yaml
docker-build:
  stage: docker-build
  image: docker:24.0.5
  variables:
    DOCKER_HOST: unix:///var/run/docker.sock
```

This job:
- Uses Docker image to run commands
- Connects to Docker socket for building images
- Only runs on the `main` branch
- Requires a runner with `docker` tag

#### Deploy Job
```yaml
deploy:
  stage: deploy
  when: manual
  dependencies:
    - docker-build
```

The deployment job:
- Runs manually for safety
- Depends on successful build
- Stops old containers and starts new ones
- Includes health checking

## Step 4: Deployment Process

### What Happens During Deployment

1. **Stop Existing Container**: Safely stops the running application
2. **Remove Old Container**: Cleans up the stopped container
3. **Start New Container**: Launches the updated application
4. **Health Check**: Verifies the application is working correctly

### Environment Variables in Production

The deployed container receives these environment variables:
- `ENVIRONMENT=production`: Identifies the environment
- `CI_COMMIT_SHA`: Git commit hash for tracking
- `CI_RUNNER_DESCRIPTION`: Information about the runner

## Step 5: Testing Your Setup

### Initial Setup
1. Push your code to GitLab:
```bash
git add .
git commit -m "Add CI/CD pipeline configuration"
git push origin main
```

2. The pipeline will automatically trigger and build your image

3. Manually trigger the deployment from GitLab's CI/CD > Pipelines page

### Verifying Deployment

After deployment, you can verify your application is running:

```bash
# Check container status
docker ps | grep python-cicd-app

# Test the application
curl http://localhost:8000/health

# View application logs
docker logs python-cicd-app
```

## Common Use Cases

### Automatic Deployments

To make deployments automatic, remove the `when: manual` line from the deploy job:

```yaml
deploy:
  stage: deploy
  # Remove this line: when: manual
```

### Multiple Environments

You can create separate deployment jobs for different environments:

```yaml
deploy-staging:
  stage: deploy
  script:
    # Deployment script for staging
  environment:
    name: staging
    url: http://staging.localhost:8000

deploy-production:
  stage: deploy
  script:
    # Deployment script for production
  environment:
    name: production
    url: http://localhost:8000
  when: manual
```

## Monitoring and Maintenance

### Checking Pipeline Status

Monitor your pipelines in GitLab:
1. Go to your project
2. Navigate to CI/CD > Pipelines
3. Click on any pipeline to see detailed logs

### Container Health

The Dockerfile includes a built-in health check that runs every 30 seconds:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Log Monitoring

View application logs:
```bash
# Real-time logs
docker logs -f python-cicd-app

# Recent logs
docker logs --tail 100 python-cicd-app
```

## Troubleshooting

### Pipeline Fails to Start
- Check if your runner is active in GitLab Settings > CI/CD > Runners
- Verify the runner has the correct tags (`docker`)

### Build Fails
- Check the pipeline logs in GitLab
- Ensure your Dockerfile and requirements.txt are correct
- Verify Docker is running on the runner machine

### Deployment Issues
- Check if port 8000 is available
- Verify the application starts correctly locally
- Review container logs for errors

### Health Check Failures
- Ensure your application has a `/health` endpoint
- Check if the application is listening on the correct port
- Verify the health endpoint returns HTTP 200

## Security Considerations

### Docker Socket Access
This setup uses Docker socket binding (`/var/run/docker.sock`) which gives the runner access to the host's Docker daemon. While convenient, be aware that:
- Containers can potentially access other containers
- Only use trusted code in your pipelines
- Consider using Docker-in-Docker for higher security in production

### Environment Variables
- Never commit secrets to your repository
- Use GitLab's CI/CD variables for sensitive data
- Mark sensitive variables as "Protected" and "Masked"

## Conclusion

You now have a working GitLab CI/CD pipeline that:
- Automatically builds Docker images on code changes
- Deploys applications with health checking
- Provides monitoring and logging capabilities
- Uses caching for improved performance

This setup provides a solid foundation for continuous deployment. You can extend it by adding testing stages, multiple environments, or notification systems based on your needs.

The pipeline ensures your application is always deployed with the latest code while maintaining reliability through health checks and manual deployment approval for production.


Consider these enhancements for your CI/CD pipeline:
- Add automated linting and testing stages before deployment stage
- Implement blue-green deployments for zero downtime
- Set up monitoring and alerting

Remember to always test your pipeline changes in a non-production environment first!
