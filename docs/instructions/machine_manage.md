## 本地机器别名配置
新注册机器统一使用宿主机 IP 作为机器别名：

| 宿主机 IP | 机器别名 |
| --- | --- |
| `90.90.97.4` | `90.90.97.4` |
| `90.90.97.15` | `90.90.97.15` |

`machine_add.py` 当前默认将 `--host` 的宿主机 IP 直接作为 alias，不再按 `/24` 网段转换别名。已有 inventory 记录继续使用原 alias；如需非 IP 命名，可显式传入 `--alias`。

注册入口示例：

```powershell
py -3 .agents/skills/machine-management/scripts/machine_add.py `
  --host 90.90.97.4 `
  --image main
```

## 容器配置
优先看机器内自带镜像：docker images
看到nightly-main-xx直接用
`dev-26.1.0.cann9.1.0.day20260810-800I-A3-py311-Ubuntu24.04-lts-aarch64`也可以，注意看日期
容器名设置xrs_vllm_main

docker run --name xrs_vllm_main -it -d --net=host --shm-size=500g \
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
    vllm-ascend:dev-26.1.0.cann9.1.0.day20260810-800I-A3-py311-Ubuntu24.04-lts-aarch64

## 机器

## 模型权重
### 97.15, 97.44： /mnt/a800_weights
Qwen3.6-35B-A3B
Qwen3-30B-A3B
DeepSeek-V4-Flash-w4a8
GLM-5.2-w4a8
MiniMax-M3

### 17.111 /mnt/weight
Qwen3-30B-A3B
Qwen3.5-35B-A3B
DeepSeek-V4-Flash


查看/mnt/a800_weights
和/mnt/share/weights
/mnt/weights
这些路径for 权重加载。