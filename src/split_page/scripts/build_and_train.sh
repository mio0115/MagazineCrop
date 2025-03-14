# !/bin/sh

IMAGE_NAME="project/mag_crop"
IMAGE_TAG="sp_pg_train"
CONTAINER_NAME="sp_pg_container"
PATH_TO_PROJECT="/home/daniel/cv_project/magazine_crop"
WORKDIR="/workspace"

LEARNING_RATE=1e-3
LEARNING_RATE_BACKBONE=1e-4
AUGMENT_FACTOR=10
EPOCHS=10
BATCH_SIZE=12
RESUME_FROM="sp_pg_model.pth"
DEVICE="cuda"
SAVE_AS="sp_pg_model.pth"
CLS_NUM=20
RESUME_FROM="sp_pg_model.pth"
SHARED_MEM_SIZE="200g"
DATALOADER_WORKERS=3
ACCUM_STEPS=1
MODULE_NAME="train"
CKPT_DIR="./checkpoints"
TRAIN=""
RESUME=""
NO_SAVE=""
MIXED_PRECISION=""

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
        --device) DEVICE="$2"; shift;;
        --save-as) SAVE_AS="$2"; shift;;
        --resume-from) RESUME_FROM="$2"; shift;;
        --shm) SHARED_MEM_SIZE="$2"; shift;;
        --dataloader-workers) DATALOADER_WORKERS="$2"; shift;;
        -ckpt-dir) CKPT_DIR="$2"; shift;;
        --checkpoint-dir) CKPT_DIR="$2"; shift;;
        -accum-steps) ACCUM_STEPS="$2"; shift;;
        --accumulation-steps) ACCUM_STEPS="$2"; shift;;
        --module-name) MODULE_NAME="$2"; shift;;
        --train) TRAIN="--train" ;;
        --resume) RESUME="--resume" ;;
        --no-save) NO_SAVE="--no-save"; shift;;
        -mp) MIXED_PRECISION="--mixed-precision"; shift;;
        --mixed-precision) MIXED_PRECISION="--mixed-precision"; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # Move to the next argument
done

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f $(pwd)/src/split_page/train.Dockerfile .
docker image prune -f

if [[ $? -ne 0 ]]; then
    echo "Docker build failed. Exiting."
    exit 1
fi

echo "Start Docker container..."

docker run --rm -it \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:${WORKDIR}/data \
        -v $(pwd)/src/split_page/checkpoints:${WORKDIR}/checkpoints \
        --gpus all \
        --shm-size=${SHARED_MEM_SIZE} \
        ${IMAGE_NAME}:${IMAGE_TAG} \
        python -m src.split_page.${MODULE_NAME} \
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
            --accumulation-steps=${ACCUM_STEPS} \
            ${TRAIN} \
            ${RESUME} \
            ${NO_SAVE} \
            ${MIXED_PRECISION} \

if [ $? -ne 0 ]; then
    echo "Docker run failed. Exiting."
    exit 1
fi