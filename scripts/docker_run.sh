#!/usr/bin/env bash

IMAGES_ID=$1
NAME=$2
if [ $# -ne 2 ]; then
    echo "error: need one argument describing your container name."
    exit 1
fi

DEVICE_ARGS=(
    --device=/dev/davinci_manager
    --device=/dev/hisi_hdc
)
if [ -e /dev/devmm_svm ]; then
    DEVICE_ARGS+=(--device=/dev/devmm_svm)
fi

docker run --name ${NAME} -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    "${DEVICE_ARGS[@]}" \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /dl:/dl \
    -v /data:/data \
    -v /data1:/data1 \
    -v /workspace:/workspace \
    -v /tmp:/tmp \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    ${IMAGES_ID}

docker run --name xrs_vllm_main -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    "${DEVICE_ARGS[@]}" \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /dl:/dl \
    -v /data:/data \
    -v /data1:/data1 \
    -v /workspace:/workspace \
    -v /tmp:/tmp \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    vllm-ascend:dev-26.1.0.day20260806-A5-py311-openEuler24.03-lts-aarch64
