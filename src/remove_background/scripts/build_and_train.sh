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
BATCH_SIZE=12
RESUME_FROM="model_weights.pth"
DEVICE="cuda"
SAVE_AS="rm_bg_unetpp.pth"
CLS_NUM=20
SHARED_MEM_SIZE="200g"
RESUME_FROM="rm_bg_unetpp_pretrained.pth"
CKPT_DIR="./checkpoints"
MODULE_NAME="train"
DATALOADER_WORKERS=3
EDGE_SIZE=640
TRAIN=""
RESUME=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --learning-rate) LEARNING_RATE="$2"; shift ;;
        -lr) LEARNING_RATE="$2"; shift ;;
        --lr-backbone) LEARNING_RATE_BACKBONE="$2"; shift;;
        --augment-factor) AUGMENT_FACTOR="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        -e) EPOCHS="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        -bs) BATCH_SIZE="$2"; shift;;
        --shm) SHARED_MEM_SIZE="$2"; shift;;
        --resume) RESUME="--resume" ;;
        --device) DEVICE="$2"; shift;;
        --save-as) SAVE_AS="$2"; shift;;
        --resume-from) RESUME_FROM="$2"; shift;;
        --train) TRAIN="--train" ;;
        --resume) RESUME="--resume" ;;
        --dataloader-workers) DATALOADER_WORKERS="$2"; shift;;
        -ckpt-dir) CKPT_DIR="$2"; shift;;
        --checkpoint-dir) CKPT_DIR="$2"; shift;;
        --module-name) MODULE_NAME="$2"; shift;;
        --edge-size) EDGE_SIZE="$2"; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # Move to the next argument
done

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f $(pwd)/src/remove_background/train.Dockerfile .
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
        --shm-size=${SHARED_MEM_SIZE} \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python -m src.remove_background.${MODULE_NAME} \
            --learning-rate=${LEARNING_RATE} \
            --lr-backbone=${LEARNING_RATE_BACKBONE} \
            --augment-factor=${AUGMENT_FACTOR} \
            --epochs=${EPOCHS} \
            --batch-size=${BATCH_SIZE} \
            --device=${DEVICE} \
            --save-as=${SAVE_AS} \
            --resume-from=${RESUME_FROM} \
            --dataloader-workers=${DATALOADER_WORKERS} \
            --checkpoint-dir=${CKPT_DIR} \
            --edge-size=${EDGE_SIZE} \
            ${TRAIN} \
            ${RESUME} \

if [ $? -ne 0 ]; then
    echo "Docker run failed. Exiting."
    exit 1
fi