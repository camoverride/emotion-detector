# This assumes that a kube cluster has already been set up

# Set environment variables
export PROJECT_ID=face-app-275621
export PROJECT_NAME=face-app
export KUBE_NAME=face-app-web 

if [ -z "$PROJECT_VERSION" ]
then
    echo "No project version found, aborting!"
else
    # Increment the project version.
    MAJOR_VERSION=$(echo $PROJECT_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $PROJECT_VERSION | cut -d. -f2-)
    ZERO=0.
    MINOR_VERSION=$ZERO$MINOR_VERSION
    MINOR_VERSION=$(bc <<< "$MINOR_VERSION + 0.01")
    PROJECT_VERSION=$(bc <<< "$MAJOR_VERSION + $MINOR_VERSION")

    export PROJECT_VERSION=$PROJECT_VERSION
    echo Creating Version $PROJECT_VERSION

    # Build it, push it, and deploy it.
    docker build -t gcr.io/${PROJECT_ID}/${PROJECT_NAME}:v${PROJECT_VERSION} .
    docker push gcr.io/${PROJECT_ID}/${PROJECT_NAME}:v${PROJECT_VERSION}
    kubectl set image deployment/${KUBE_NAME} ${PROJECT_NAME}=gcr.io/${PROJECT_ID}/${PROJECT_NAME}:v${PROJECT_VERSION}
fi

# Cleanup: remove all the <none> images
docker rmi $(docker images -f "dangling=true" -q)

# Show the kube deployment
kubectl get service
