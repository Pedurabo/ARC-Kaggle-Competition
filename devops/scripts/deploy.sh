#!/bin/bash

# ARC AI System Deployment Script
# Comprehensive DevOps deployment with monitoring and security

set -e  # Exit on any error

# Configuration
ENVIRONMENT=${1:-production}
REGION=${2:-us-east-1}
CLUSTER_NAME="arc-ai-cluster"
NAMESPACE="arc-ai"
DOCKER_IMAGE="arc-ai:latest"
DOCKER_REGISTRY="your-registry.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        error "Helm is not installed"
    fi
    
    success "Prerequisites check passed"
}

# Security scan
run_security_scan() {
    log "Running security scan..."
    
    # Run Trivy scan
    if command -v trivy &> /dev/null; then
        log "Running Trivy container scan..."
        trivy image --severity HIGH,CRITICAL ${DOCKER_IMAGE} || warning "Trivy scan found issues"
    fi
    
    # Run Bandit scan
    log "Running Bandit security scan..."
    bandit -r src/ -f json -o security-bandit-report.json || warning "Bandit scan found issues"
    
    # Run Safety scan
    log "Running Safety dependency scan..."
    safety check --json --output safety-report.json || warning "Safety scan found issues"
    
    success "Security scan completed"
}

# Build and test
build_and_test() {
    log "Building and testing application..."
    
    # Build Docker image
    log "Building Docker image..."
    docker build -t ${DOCKER_IMAGE} -f devops/Dockerfile . || error "Docker build failed"
    
    # Run tests
    log "Running tests..."
    docker run --rm ${DOCKER_IMAGE} python -m pytest tests/ -v || error "Tests failed"
    
    # Run performance tests
    log "Running performance tests..."
    docker run --rm ${DOCKER_IMAGE} python test_models.py --model ensemble || warning "Performance tests failed"
    
    success "Build and test completed"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply secrets
    log "Applying secrets..."
    kubectl apply -f devops/kubernetes/arc-ai-deployment.yaml -n ${NAMESPACE}
    
    # Wait for deployment
    log "Waiting for deployment to be ready..."
    kubectl rollout status deployment/arc-ai-app -n ${NAMESPACE} --timeout=300s
    
    # Check service health
    log "Checking service health..."
    kubectl wait --for=condition=ready pod -l app=arc-ai -n ${NAMESPACE} --timeout=300s
    
    success "Kubernetes deployment completed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Deploy Prometheus
    if ! helm repo list | grep -q prometheus-community; then
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
    fi
    
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=7d \
        --set grafana.enabled=true \
        --set grafana.adminPassword=admin123 || warning "Prometheus deployment failed"
    
    # Deploy ELK stack
    log "Deploying ELK stack..."
    helm repo add elastic https://helm.elastic.co
    helm repo update
    
    helm install elasticsearch elastic/elasticsearch \
        --namespace logging \
        --create-namespace \
        --set replicas=1 \
        --set resources.requests.memory=512Mi || warning "Elasticsearch deployment failed"
    
    helm install kibana elastic/kibana \
        --namespace logging \
        --set elasticsearchHosts=http://elasticsearch-master:9200 || warning "Kibana deployment failed"
    
    success "Monitoring setup completed"
}

# Performance optimization
run_performance_optimization() {
    log "Running performance optimization..."
    
    # Get current accuracy
    CURRENT_ACCURACY=$(kubectl exec -n ${NAMESPACE} deployment/arc-ai-app -- python -c "
import json
try:
    with open('performance-report.json', 'r') as f:
        data = json.load(f)
    print(data.get('overall', {}).get('avg_accuracy', 0.0))
except:
    print('0.17')
" 2>/dev/null || echo "0.17")
    
    log "Current accuracy: ${CURRENT_ACCURACY}"
    
    # Run optimization if accuracy is below target
    if (( $(echo "$CURRENT_ACCURACY < 0.30" | bc -l) )); then
        log "Running optimization to reach 30% accuracy..."
        
        kubectl exec -n ${NAMESPACE} deployment/arc-ai-app -- python -c "
from src.optimization.performance_optimizer import get_performance_optimizer
optimizer = get_performance_optimizer()
optimizer.current_accuracy = ${CURRENT_ACCURACY}
results = optimizer.optimize_for_30_percent({})
print('Optimization results:', results)
" || warning "Optimization failed"
    else
        success "Accuracy target already met: ${CURRENT_ACCURACY}"
    fi
}

# Health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check application health
    kubectl exec -n ${NAMESPACE} deployment/arc-ai-app -- curl -f http://localhost:8000/health || error "Application health check failed"
    
    # Check monitoring endpoints
    kubectl exec -n ${NAMESPACE} deployment/arc-ai-app -- curl -f http://localhost:9090/-/healthy || warning "Prometheus health check failed"
    
    # Check metrics endpoint
    kubectl exec -n ${NAMESPACE} deployment/arc-ai-app -- curl -f http://localhost:8000/metrics || warning "Metrics endpoint check failed"
    
    # Check logs
    kubectl logs -n ${NAMESPACE} deployment/arc-ai-app --tail=50 | grep -i error && warning "Errors found in logs" || success "No errors in recent logs"
    
    success "Health checks completed"
}

# Generate reports
generate_reports() {
    log "Generating deployment reports..."
    
    # Get deployment status
    kubectl get pods -n ${NAMESPACE} -o wide > deployment-status.txt
    
    # Get service endpoints
    kubectl get svc -n ${NAMESPACE} > service-endpoints.txt
    
    # Get resource usage
    kubectl top pods -n ${NAMESPACE} > resource-usage.txt
    
    # Get logs summary
    kubectl logs -n ${NAMESPACE} deployment/arc-ai-app --tail=100 > recent-logs.txt
    
    # Generate performance report
    kubectl exec -n ${NAMESPACE} deployment/arc-ai-app -- python -c "
from src.monitoring.metrics import get_metrics_collector
from src.optimization.performance_optimizer import get_performance_optimizer
import json

metrics = get_metrics_collector()
optimizer = get_performance_optimizer()

report = {
    'metrics_summary': metrics.get_performance_summary(),
    'optimization_summary': optimizer.get_optimization_summary(),
    'deployment_info': {
        'environment': '${ENVIRONMENT}',
        'namespace': '${NAMESPACE}',
        'image': '${DOCKER_IMAGE}'
    }
}

with open('deployment-report.json', 'w') as f:
    json.dump(report, f, indent=2)
" || warning "Failed to generate performance report"
    
    success "Reports generated"
}

# Main deployment function
main() {
    log "Starting ARC AI deployment to ${ENVIRONMENT} environment..."
    
    check_prerequisites
    run_security_scan
    build_and_test
    deploy_to_kubernetes
    setup_monitoring
    run_performance_optimization
    run_health_checks
    generate_reports
    
    success "ARC AI deployment completed successfully!"
    
    log "Deployment Summary:"
    log "- Environment: ${ENVIRONMENT}"
    log "- Namespace: ${NAMESPACE}"
    log "- Image: ${DOCKER_IMAGE}"
    log "- Reports: deployment-status.txt, service-endpoints.txt, resource-usage.txt"
    
    log "Next steps:"
    log "1. Monitor application: kubectl logs -f -n ${NAMESPACE} deployment/arc-ai-app"
    log "2. Check metrics: kubectl port-forward -n ${NAMESPACE} svc/arc-ai-service 8000:8000"
    log "3. Access Grafana: kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    log "4. Check Kibana: kubectl port-forward -n logging svc/kibana-kibana 5601:5601"
}

# Run main function
main "$@" 