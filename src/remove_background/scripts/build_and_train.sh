# !/bin/sh

IMAGE_NAME="project/mag_crop"
IMAGE_TAG="rm_bg_train"
CONTAINER_NAME="rm_bg_container"
PATH_TO_PROJECT="/home/daniel/cv_project/magazine_crop"
WORKDIR="/workspace"

LEARNING_RATE=1e-3
LEARNING_RATE_BACKBONE=1e-4
AUGMENT_FACTOR=10
EPOCHS=10
BATCH_SIZE=16
RESUME_FROM="model_weights.pth"
DEVICE="cuda"
SAVE_AS="model.weights.pth"
CLS_NUM=20
TRAIN=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        -lr) LEARNING_RATE="$2"; shift ;;
        --lr_backbone) LEARNING_RATE_BACKBONE="$2"; shift;;
        --augment_factor) AUGMENT_FACTOR="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        -e) EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        -bs) BATCH_SIZE="$2"; shift;;
        --resume) RESUME="--resume" ;;
        --device) DEVICE="$2"; shift;;
        --save_as) SAVE_AS="$2"; shift;;
        --train) TRAIN="--train" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # Move to the next argument
done

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f $(pwd)/src/remove_background/dockerfile_train .
docker image prune -f

if [[ $? -ne 0 ]]; then
    echo "Docker build failed. Exiting."
    exit 1
fi

echo "Start Docker container..."

docker run --rm -it \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:${WORKDIR}/data \
        -v $(pwd)/src/remove_background/checkpoints:${WORKDIR}/checkpoints \
        --gpus all \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python -m src.remove_background.train \
            --learning_rate=${LEARNING_RATE} \
            --lr_backbone=${LEARNING_RATE_BACKBONE} \
            --augment_factor=${AUGMENT_FACTOR} \
            --epochs=${EPOCHS} \
            --batch_size=${BATCH_SIZE} \
            --device=${DEVICE} \
            --save_as=${SAVE_AS} \
            ${TRAIN}

if [ $? -ne 0 ]; then
    echo "Docker run failed. Exiting."
    exit 1
fi