# 90.90.97.4 / xrs_vllm_main 远程配置与代码同步记录

更新时间：2026-08-07

## 1. 目标配置

| 项目 | 配置 |
| --- | --- |
| 宿主机 IP | `90.90.97.4` |
| 机器别名 | `90.90.97.4` |
| 容器名 | `xrs_vllm_main` |
| 容器 SSH 端口 | `46000` |
| 容器工作目录 | `/home/x50063850/vllm-ascend-workspace` |
| 机器类型 | `A3` |
| SoC | `ascend910b1` |
| 镜像 | `vllm-ascend:dev-26.1.0.cann9.1.0.day20260801-800I-A3-py311-Ubuntu24.04-lts-aarch64` |

注意：`xrs_vllm_main` 已经被 `80.5.17.111` 使用为机器别名，因此本机使用 IP `90.90.97.4` 作为唯一 alias；容器名仍是 `xrs_vllm_main`。

配置落盘位置：

- `.vaws-local/machine-inventory.json`
- `.remote-dev/endpoints.local.json`
- `.vaws-local/remote-code-parity/install-consents.json`

## 2. 容器环境配置

目标容器原本正在运行，但没有可用的容器 SSH。已在容器内配置：

- `/etc/ssh/sshd_vaws_config`
- `/root/.ssh/authorized_keys`
- `/etc/profile.d/vaws-ascend-env.sh`
- `/etc/vaws/container-info.json`
- `/etc/pip.conf`

本机旧的 `[90.90.97.4]:46000` known_hosts 记录属于已退出的旧容器，已清理并接受当前 `xrs_vllm_main` 的新 host key。

手工 NPU smoke 结果：

```text
torch: 2.10.0+cpu
torch_npu: 2.10.0.post4
torch.ones(1).npu() -> [1.0]
```

## 3. 只同步代码、不编译

首次使用该容器需要记录本地代码同步授权：

```powershell
py -3 .agents/skills/remote-code-parity/scripts/install_consent.py set-sync-mode `
  --server-name 90.90.97.4 `
  --container-identity 'xrs_vllm_main@/home/x50063850/vllm-ascend-workspace' `
  --sync-mode local `
  --allow-first-install `
  --approved-by-user
```

只物化代码的同步命令：

```powershell
py -3 .agents/skills/remote-code-parity/scripts/parity_sync.py `
  --machine 90.90.97.4 `
  --apply-mode materialize
```

`materialize` 模式包含：

1. 从本地工作树创建 synthetic Git snapshot。
2. 将 root、`vllm`、`vllm-ascend` 及递归子模块推送到容器本地 mirror。
3. 将 snapshot 物化到 `/home/x50063850/vllm-ascend-workspace`。
4. 校验远端 runtime commit 与 snapshot commit 一致。

该模式跳过：

- `pip install`。
- vLLM editable 安装。
- vllm-ascend 自定义算子编译。
- 依赖安装和 runtime marker 写入。

## 4. 本次同步结果

```text
status: materialized
apply_mode: materialize
reinstall: skipped-by-apply-mode
workspace_id: vllm-ascend-workspace-edbc1992
snapshot_id: 20260807T084542Z-bd717f5e
manifest: /root/.cache/vaws/remote-code-parity/workspaces/vllm-ascend-workspace-edbc1992/manifests/20260807T084542Z-bd717f5e.json
```

已验证的 runtime commits：

```text
root:        faaf8abbeb4e25ea618e40da47bf945fbb0fd8a6
vllm:        9374f773a0565c7089530d7dbe759bb64bb46613
vllm-ascend: 29295b5a33db7691537c9bb0f0dd94f45813f7af
catlass:     23364ef2b039c8b1725a541376966209f147e41a
googletest:  f6c54a781bf4d764f7f9e75c9e3d2eeec7f8718a
```

结论：当前本地代码已经同步到 `90.90.97.4` 的 `xrs_vllm_main`，本次没有执行编译或安装。后续如果需要真正运行服务，应先单独执行完整 `install` parity，并确认依赖和自定义算子构建完成。
