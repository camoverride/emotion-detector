# Builds and deploys a new docker image with tensorflow serving.

# Set environment variables
export PROJECT_ID=emotion-model-1-276322
export PROJECT_NAME=emotion-model-1
export KUBE_NAME=emotion-model-server
export PROJECT_VERSION=0.1

if [ -z "$PROJECT_VERSION" ]
then
    echo "No project version found, aborting!"
else
    # Build it, push it, and deploy it.
    docker build -t gcr.io/${PROJECT_ID}/${PROJECT_NAME}:v${PROJECT_VERSION} .
    docker push gcr.io/${PROJECT_ID}/${PROJECT_NAME}:v${PROJECT_VERSION}
    kubectl set image deployment/${KUBE_NAME} ${PROJECT_NAME}=gcr.io/${PROJECT_ID}/${PROJECT_NAME}:v${PROJECT_VERSION}
fi

# Cleanup: remove all the <none> images
docker rmi $(docker images -f "dangling=true" -q)

# Show the kube deployment
kubectl get service
