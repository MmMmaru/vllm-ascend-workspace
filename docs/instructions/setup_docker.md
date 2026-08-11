## 容器配置
优先看机器内自带镜像：docker images
看到nightly-main-xx直接用

docker run --name xrs_vllm_main1 -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
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
    7a85b5e7f489

## 机器