# 讲解 machine 注册流程

> 讲解一下machine注册的脚本、流程

<details><summary>6 previous messages</summary>

> 我会使用 `concise-code-explanation` 技能，结合实际源码和行号，梳理 machine 注册脚本的调用链、状态文件变化与远端验证流程。先读取仓库约束和相关脚本，再给出“总—分—总”的说明。
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content .agents/skills/concise-code-explanation/SKILL.md; Get-Content .agents/AGENTS.md; Get-Content Contributing.md; Get-Content CONTEXT.md`
>
> </details>
>
> 已确认这是仓库内 `machine-management` 工作流；我还会读取该技能的说明以及 machine 管理文档，重点区分“注册到 inventory”和“创建/修复远端容器”这两个容易混淆的阶段。
>
> <details><summary>Ran 9 commands</summary>
>
> - Ran `Get-Content .agents/skills/machine-management/SKILL.md -Encoding UTF8; Get-Content docs/instructions/machine_manage.md -Encoding UTF8; rg -n "^(def |class |if __name__|parser|MACHINE|inventory|register|add|probe|ready|container|ssh|host|port|machine)" .agents/skills/machine-management/scripts docs/instructions/machine_manage.md .vaws-local/machine-inventory.json`
> - Ran `$files = @('.agents/skills/machine-management/scripts/machine_add.py','.agents/skills/machine-management/scripts/_workflow_common.py','.agents/skills/machine-management/scripts/inventory.py','.agents/scripts/workspace_profile.py','.vaws-local/machine-inventory.json','.vaws-local/machine-profile.json'); foreach ($f in $files) { Write-Output "===== $f ====="; if (Test-Path $f) { $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if (($f -like '*machine_add.py' -and (($i -le 180) -or ($i -ge 180 -and $i -le 390))) -or ($f -like '*_workflow_common.py' -and (($i -ge 200 -and $i -le 430) -or ($i -ge 470 -and $i -le 690) -or ($i -ge 780 -and $i -le 930))) -or ($f -like '*inventory.py' -and (($i -le 180) -or ($i -ge 238 -and $i -le 270))) -or ($f -like '*workspace_profile.py' -and ($i -le 220)) -or ($f -like '*.json')) { '{0,4}: {1}' -f $i, $_ } } } }`
> - Ran `$f='.agents/skills/machine-management/scripts/_workflow_common.py'; $ranges=@(@(217,380),@(419,620),@(620,850)); foreach($r in $ranges){ Write-Output "===== $f:$($r[0])-$($r[1]) ====="; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if($i -ge $r[0] -and $i -le $r[1]) {'{0,4}: {1}' -f $i,$_} } }`
> - Ran `$f='.agents/skills/machine-management/scripts/_workflow_common.py'; $ranges=@(@(217,380),@(419,620),@(620,850)); foreach($r in $ranges){ Write-Output "===== ${f}:$($r[0])-$($r[1]) ====="; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if($i -ge $r[0] -and $i -le $r[1]) {'{0,4}: {1}' -f $i,$_} } }`
> - Ran `$f='.agents/skills/machine-management/scripts/_workflow_common.py'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if($i -ge 847 -and $i -le 890) {'{0,4}: {1}' -f $i,$_} }; Write-Output '===== relevant manage_machine command entry ====='; $f2='.agents/skills/machine-management/scripts/manage_machine.py'; $i=0; Get-Content $f2 -Encoding UTF8 | ForEach-Object { $i++; if(($i -ge 977 -and $i -le 1038) -or ($i -ge 1223 -and $i -le 1415) -or ($i -ge 1563 -and $i -le 1705) -or ($i -ge 2093 -and $i -le 2178) -or ($i -ge 2490 -and $i -le 2605)) {'{0,4}: {1}' -f $i,$_} }`
> - Ran `rg -n "npu-smi|free_port|docker|apt|sshd|container-info|host-info|pip.conf|run_with_progress|selected_image|container_exists|replace_on_image_change|already" .agents/skills/machine-management/scripts/manage_machine.py`
> - Ran `$f='.agents/skills/machine-management/scripts/manage_machine.py'; $ranges=@(@(1031,1221),@(1420,1558),@(1730,1968),@(1990,2080),@(2178,2220)); foreach($r in $ranges){ Write-Output "===== ${f}:$($r[0])-$($r[1]) ====="; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if($i -ge $r[0] -and $i -le $r[1]) {'{0,4}: {1}' -f $i,$_} } }`
> - Ran `rg -n "derive_network_segment_alias|choose_alias\(|--alias|machine_add.py|machine_verify.py|machine_repair.py|machine_remove.py" .agents/skills/machine-management docs .agents/scripts | Select-Object -First 120; $f='.agents/skills/machine-management/scripts/machine_add.py'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if($i -ge 76 -and $i -le 190) {'{0,4}: {1}' -f $i,$_} }`
> - Ran `$f='.agents/skills/machine-management/scripts/manage_machine.py'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if($i -ge 166 -and $i -le 270) {'{0,4}: {1}' -f $i,$_} }; Write-Output '===== machine verify wrapper ====='; $f2='.agents/skills/machine-management/scripts/machine_verify.py'; $i=0; Get-Content $f2 -Encoding UTF8 | ForEach-Object { $i++; if($i -le 100) {'{0,4}: {1}' -f $i,$_} }; Write-Output '===== machine repair entry ====='; $f3='.agents/skills/machine-management/scripts/machine_repair.py'; $i=0; Get-Content $f3 -Encoding UTF8 | ForEach-Object { $i++; if(($i -le 130) -or ($i -ge 250 -and $i -le 315)) {'{0,4}: {1}' -f $i,$_} }`
>
> </details>
>
> 主流程已经定位：`machine_add.py` 只是编排器，真正的远端探测、Docker/SSH 容器初始化和 NPU smoke test 在 `manage_machine.py` 生成的远端 Bash 脚本中；本地状态写入由 `_workflow_common.py` + `inventory.py` 完成。接下来我核对验收约束和当前 `xrs_vllm_main` 记录，避免把“已有容器接管”讲成“新建容器”。
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content .agents/skills/machine-management/references/behavior.md -Encoding UTF8; Get-Content .agents/skills/machine-management/references/acceptance.md -Encoding UTF8; Get-Content .agents/skills/machine-management/references/command-recipes.md -Encoding UTF8`
>
> </details>
</details>

## 一、整体流程

machine 注册的入口是 `machine_add.py`。它负责把一台远端 NPU 主机接入当前 workspace，并最终确认：

1. 本地可以用 SSH Key 登录宿主机；
2. 本地可以直接 SSH 到容器；
3. 容器内 `torch`、`torch_npu` 可以正常导入并执行 NPU tensor；
4. 注册信息写入 `.vaws-local/machine-inventory.json`。

整体数据流：

```text
命令参数
  |
  v
machine_add.py
  |
  +--> machine-profile.json
  |      生成 namespace / 默认容器名
  |
  +--> machine-inventory.json
  |      查找已有机器，决定新增还是接管/修复
  |
  +--> 宿主机 SSH Key 检查
  |      必要时只使用一次密码安装公钥
  |
  +--> 远端 host probe
  |      Docker / NPU 设备 / npu-smi / 空闲端口 / 已有容器
  |
  +--> 远端 container bootstrap
  |      docker run 或复用容器
  |      安装 SSH、配置环境、启动专用 sshd
  |
  +--> inventory upsert
  |      保存机器、容器、镜像、硬件信息
  |
  +--> mesh 同步
  |
  +--> final verify
         宿主机 SSH
         容器 SSH
         torch + torch_npu smoke
```

入口参数和主流程位于 [`machine_add.py`](./.agents/skills/machine-management/scripts/machine_add.py:40)。

---

## 二、注册脚本的主要阶段

### 1. 初始化本地 profile

脚本首先读取或创建：

```text
.vaws-local/machine-profile.json
```

当前 profile 是：

```json
{
  "machine_username": "agent27728",
  "container_name": "vaws-agent27728"
}
```

这个 username 用作：

- workspace namespace；
- 新容器默认名称；
- 多台机器之间的名称隔离。

如果 profile 不存在，脚本不会静默生成，而是返回 `needs_input`，要求指定：

```powershell
py -3 .agents/skills/machine-management/scripts/machine_add.py `
  --host <IP> `
  --image main `
  --machine-username alice123
```

相关逻辑在 [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:338)。

---

### 2. 查找 inventory，决定新增还是接管

脚本会同时按 host IP 和 alias 查找：

```text
.vaws-local/machine-inventory.json
```

如果找到已有记录，就复用其中的：

- alias；
- 宿主机用户名和 SSH 端口；
- 容器名称；
- 容器 SSH 端口；
- workdir；
- namespace。

因此，当前 inventory 中的：

```json
{
  "host": {
    "ip": "90.90.97.4"
  },
  "container": {
    "name": "xrs_vllm_main",
    "ssh_port": 46000
  },
  "created_by_skill": false,
  "managed_by_skill": true
}
```

表示这个 `xrs_vllm_main` 容器原本不是脚本创建的，但已经被 workspace 接管管理。

如果已有机器已经满足：

- host SSH；
- container SSH；
- smoke；
- 镜像匹配；

则直接返回 `already-ready`，不会重复创建容器。逻辑位于 [`machine_add.py`](./.agents/skills/machine-management/scripts/machine_add.py:93)。

---

### 3. 镜像选择

新机器必须显式指定镜像：

```text
--image rc
--image main
--image stable
--image <具体镜像:tag>
```

禁止默认使用：

```text
auto
latest
没有 tag 的镜像仓库
```

硬件相关镜像会自动追加后缀：

```text
A2   -> 基础 tag
A3   -> 基础 tag-a3
310P -> 基础 tag-310p
```

例如：

```text
main
  -> quay.nju.edu.cn/ascend/vllm-ascend:main-a3
  -> quay.io/ascend/vllm-ascend:main-a3
```

已有 inventory 中的固定镜像可以复用，但如果是旧的、模糊的或 moving tag，脚本会先要求重新选择镜像。镜像匹配逻辑在 [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:217)。

---

### 4. 宿主机 SSH Key 注册

脚本先检查本地 SSH Key 是否存在，然后检查：

```text
本地 -> 宿主机:22
```

如果 Key 登录已经成功，就不会使用密码。

如果 Key 登录失败，并且提供了密码，则执行一次性 host bootstrap：

1. 通过密码登录宿主机；
2. 将本地公钥追加到宿主机的 `authorized_keys`；
3. 再次用 Key 登录确认；
4. 后续流程全部改用 Key。

密码不会写入 inventory，也不会用于容器登录。相关逻辑在 [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:422)。

---

### 5. 远端 host probe

host probe 是只读探测，检查：

- Docker 是否存在；
- `docker info` 是否正常；
- `/dev/davinci_manager` 等 NPU 设备；
- Ascend driver 和 DCMI 路径；
- `npu-smi` 是否可用；
- `npu-smi info` 输出的 SoC；
- 当前已有的 `vaws-*` 容器；
- 高位 SSH 端口是否空闲。

机器类型检测结果类似：

```text
ascend910b1 -> A3
ascend910_9391 -> A3
310P       -> 310P
```

如果无法从 `npu-smi` 判断类型，脚本不会猜测，而是返回：

```text
需要重新执行 --machine-type A2|A3|310P
```

探测脚本由 [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:1031) 动态生成并通过宿主机 SSH 执行。

---

### 6. 创建或复用容器

容器 bootstrap 分两条路径。

#### 已有容器

如果 Docker 中已经存在记录的容器：

```text
xrs_vllm_main
```

脚本会：

- 检查容器镜像是否匹配；
- 容器停止则启动；
- 不会随意删除；
- 重新配置容器内环境和 sshd。

只有显式指定了新镜像并确认镜像变化时，才允许因镜像变化删除并重建。

#### 新建容器

新建时大致执行：

```bash
docker run \
  --name <container> \
  --network host \
  --privileged=true \
  --shm-size=500g \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -w /vllm-workspace \
  <image>
```

实际命令由 [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:1923) 中的远端脚本生成。

---

### 7. 配置容器 SSH 和运行环境

容器创建或接管后，脚本会：

1. 安装 `openssh-server` 和 `openssh-client`；
2. apt 环境先改成 NJU mirror；
3. 写入 `/etc/pip.conf`；
4. 安装本地公钥；
5. 创建 `/run/sshd`；
6. 写入 `/etc/ssh/sshd_vaws_config`；
7. 使用高位端口启动专用 sshd；
8. 禁用密码登录，只允许 Key 登录；
9. 必要时开放宿主机防火墙端口。

容器内会持久化：

```text
/etc/vaws/container-info.json
/etc/profile.d/vaws-ascend-env.sh
/etc/pip.conf
/etc/ssh/sshd_vaws_config
```

宿主机上会持久化：

```text
/etc/vaws/host-info.json
/etc/profile.d/vaws-ascend-env.sh
```

配置逻辑集中在 [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:1734)。

---

### 8. 写入 inventory

容器 bootstrap 成功后，脚本才写入 inventory。

典型记录包括：

```json
{
  "alias": "90.90.97.4",
  "namespace": "agent27728",
  "host": {
    "ip": "90.90.97.4",
    "port": 22,
    "user": "root",
    "machine_type": "A3",
    "soc": "ascend910b1"
  },
  "container": {
    "name": "xrs_vllm_main",
    "ssh_port": 46000,
    "image": "...",
    "workdir": "...",
    "machine_type": "A3"
  },
  "bootstrap_method": "ssh",
  "managed_by_skill": true
}
```

写入过程带文件锁，并使用临时文件加原子替换，避免多个 agent 并发注册时互相覆盖。核心逻辑位于 [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:792)。

---

### 9. mesh 同步

注册成功后，脚本会遍历 inventory 中其他机器：

1. 从新容器导出 mesh 公钥；
2. 从旧容器导出 mesh 公钥；
3. 双向写入对方的 `authorized_keys`；
4. 更新 `known_hosts`。

这是 best-effort 操作。mesh 失败不会否定主机和容器的基本 ready 状态。

---

### 10. 最终 readiness 验证

最终验证不是简单的端口检查，而是三层：

```text
host SSH
  +
container SSH
  +
torch / torch_npu / NPU tensor smoke
```

smoke 会在容器内执行：

```python
import torch
import torch_npu

x = torch.zeros(1, 2).npu()
```

只有三项全部成功，最终状态才是：

```json
{
  "status": "ready",
  "success": true
}
```

验证逻辑在 [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:619)，NPU smoke 脚本在 [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:2093)。

---

## 三、几个脚本的职责边界

| 脚本 | 作用 |
|---|---|
| `machine_add.py` | 完整注册、接管、修复、写 inventory、mesh、最终验证 |
| `machine_verify.py` | 只读检查，不自动修复 |
| `machine_repair.py` | 对已有记录做保守修复 |
| `machine_remove.py` | 删除 inventory 记录及其被管理的容器 |
| `_workflow_common.py` | 编排公共流程、状态转换、inventory 更新 |
| `manage_machine.py` | SSH 执行远端脚本、Docker、sshd、probe、smoke |
| `inventory.py` | inventory schema、校验、锁和原子写入 |
| `workspace_profile.py` | 管理本地 machine username 和默认容器名 |

---

## 四、当前代码的一个注意点

文档描述过基于 `/24` 网段生成别名，例如：

```text
90.90.97.4 -> 90.90.97-4
```

但当前实际代码中的 `choose_alias()` 是：

```python
explicit_alias or host
```

也就是说，当前实现默认仍使用 host IP 作为 alias；如果要使用其他别名，需要显式传入 `--alias`。这也和当前 inventory 中的 `90.90.97.4`、`90.90.97.15` 等记录一致。

总结来说，machine 注册不是单纯“写一行 inventory”，而是：

```text
本地身份
 -> 宿主机 Key SSH
 -> 宿主机硬件/Docker 探测
 -> 创建或接管容器
 -> 容器专用 SSH
 -> inventory 持久化
 -> NPU smoke
 -> ready
```

其中真正的 ready 判定核心是“宿主机 SSH + 容器 SSH + `torch_npu` NPU 执行”三者同时成功。

<details><summary>4 previous messages</summary>

> 我会把 machine 注册文档统一改成“默认使用宿主机 IP 作为 alias”，并检查 machine-management 技能文档中是否还有 `/24` 网段别名的旧描述，避免不同文档互相矛盾。此次只改文档，不改注册脚本逻辑。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content .agents/AGENTS.md -Encoding UTF8 | Select-Object -First 220; rg -n -C 3 "网段|/24|90\.90\.97-4|derive_network|network segment|host IP|host value|alias" docs .agents/skills/machine-management`
> - Ran `Get-Content docs/instructions/machine_manage.md -Encoding UTF8; Write-Output '===== relevant skill alias sections ====='; $f='.agents/skills/machine-management/SKILL.md'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if(($i -ge 118 -and $i -le 142) -or ($i -ge 160 -and $i -le 190)) {''{0,4}: {1}'' -f $i,$_} }; Write-Output '===== git status ====='; git status --short`
> - Ran `Get-Content docs/instructions/machine_manage.md -Encoding UTF8; Write-Output '===== relevant skill alias sections ====='; $f='.agents/skills/machine-management/SKILL.md'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if(($i -ge 118 -and $i -le 142) -or ($i -ge 160 -and $i -le 190)) {'{0,4}: {1}' -f $i,$_} }; Write-Output '===== git status ====='; git status --short`
>
> </details>
>
> 检查结果显示，`machine_manage.md` 和 machine-management 技能文档的主体已经是“IP 作为 alias”；我会补一条明确说明，强调当前注册脚本的默认行为是 `alias = host IP`，并不再保留 `/24` 网段别名规则。已有的未提交修改我会保留。
>
> <details><summary>Edited a file, ran a command</summary>
>
> - Wrote `./docs/instructions/machine_manage.md` (+1 -1)
> - Ran `Get-Content docs/instructions/machine_manage.md -Encoding UTF8 | Select-Object -First 14; rg -n "网段名|网段.*主机号|90\.90\.97-4|derive_network_segment_alias|network segment" docs/instructions/machine_manage.md .agents/skills/machine-management/SKILL.md .agents/skills/machine-management/references`
>
> </details>
</details>

已修改 [`machine_manage.md`](./docs/instructions/machine_manage.md:9)，明确说明：

- 默认直接使用宿主机 IP 作为 alias；
- 不再按 `/24` 网段转换；
- 只有传入 `--alias` 时才使用非 IP 别名；
- 已有 inventory 记录继续保留原 alias。

未运行测试，仅完成文档修改。 :codex-annotation{index="1"}

> 我直接改machine invetory可以吗？还有什么别的注册文件吗？我这里需要改名容器

<details><summary>2 previous messages</summary>

> 我会按 `machine-management` 的规则检查容器名的真实来源和持久化位置，重点确认：手改 inventory 是否足够、远端 Docker 容器是否需要同步改名、以及后续 `verify/repair/serve/parity` 会读取哪些字段。先只读检查，不会直接改机器或 inventory。
>
> <details><summary>Ran 6 commands</summary>
>
> - Ran `Get-Content .agents/skills/machine-management/SKILL.md -Encoding UTF8; rg -n -C 4 "container_name|container\[\"name\"\]|container\.get\(\"name\"|--name|rename|xrs_vllm_main|machine-inventory|machine-profile|container_target" .agents/skills/machine-management/scripts .agents/lib .agents/scripts docs scripts | Select-Object -First 260; Write-Output '===== inventory records with xrs_vllm_main ====='; $f='.vaws-local/machine-inventory.json'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object { $i++; if(($i -ge 20 -and $i -le 70) -or ($i -ge 44 -and $i -le 90)) {'{0,4}: {1}' -f $i,$_} }`
> - Ran `rg -n -F -C 3 'machine-inventory.json' .agents scripts docs; rg -n -F -C 3 'container["name"]' .agents/skills/machine-management/scripts; rg -n -F -C 3 'record["container"]["name"]' .agents/skills/machine-management/scripts`
> - Ran `$paths=@('.remote-dev/endpoints.local.json','.vaws-local/machine-profile.json','.vaws-local/machine-inventory.json','.machine-inventory.json'); foreach($p in $paths){ Write-Output "===== $p ====="; if(Test-Path $p){ Get-Content $p -Encoding UTF8 | Select-Object -First 120 } else { Write-Output '(absent)' } }; Write-Output '===== container-name readers ====='; rg -n -F '["container"]["name"]' .agents; rg -n -F 'container_name' .agents/skills/remote-code-parity .agents/lib .agents/skills/machine-management/scripts | Select-Object -First 160`
> - Ran `Get-ChildItem .vaws-local -Recurse -File | Select-Object -ExpandProperty FullName; $files=Get-ChildItem .vaws-local -Recurse -File -ErrorAction SilentlyContinue; foreach($f in $files){ $m=Select-String -Path $f.FullName -Pattern 'xrs_vllm_main|container_name|container' -SimpleMatch -ErrorAction SilentlyContinue; if($m){ Write-Output "===== $($f.FullName) ====="; $m | Select-Object -First 30 | ForEach-Object { "$($_.LineNumber): $($_.Line)" } } }`
> - Ran `$files=@('.vaws-local/serving/90.90.97.4.json','.vaws-local/serving/xrs_vllm_main.json','.vaws-local/serving/xrs_vllm_main1.json','.vaws-local/current-session.json'); foreach($p in $files){ Write-Output "===== $p ====="; if(Test-Path $p){ Get-Content $p -Encoding UTF8 | Select-Object -First 100 } }`
> - Ran `rg -n "def container_target|container_name = record|def upsert_machine_record|def cleanup_parity_state|container_identity|def canonical_inventory_path|def load_machine_inventory" .agents/skills/machine-management/scripts/_workflow_common.py .agents/skills/remote-code-parity/scripts/parity_sync.py .agents/lib/vaws_remote_toolbox.py .agents/lib/vaws_local_state.py; rg -n "machine.*inventory|serving.*machine|container.*name" .agents/skills/vllm-ascend-serving .agents/skills/vllm-ascend-benchmark 2>$null | Select-Object -First 100`
>
> </details>
</details>

可以直接改，但只改本地 inventory 不够。

当前 machine 注册的关键状态是：

- canonical 文件：[`.vaws-local/machine-inventory.json`](./.vaws-local/machine-inventory.json:48)
- 本地 profile：[`.vaws-local/machine-profile.json`](./.vaws-local/machine-profile.json:1)
- 远端真实容器：宿主机 Docker 中的容器名
- 远端元数据：`/etc/vaws/host-info.json` 可能记录旧容器名

`machine_add.py` 后续会直接读取 inventory 中的 `container.name`，例如 [`machine_add.py`](./.agents/skills/machine-management/scripts/machine_add.py:257)。所以只改 inventory 而不改远端 Docker，会导致 SSH、verify 或 repair 找不到容器。

建议改名流程：

```text
远端宿主机：
docker rename xrs_vllm_main <new-name>

本地：
修改 .vaws-local/machine-inventory.json:
container.name = <new-name>

之后：
machine_verify.py --machine <宿主机IP>
```

不需要修改：

- `machine-profile.json`：它只负责新机器的默认容器名，已有 inventory 会优先使用自己的容器名；
- `.remote-dev/endpoints.local.json`：它只记录 SSH host/port，不记录 Docker 容器名；端口不变时无需改；
- `.vaws-local/serving/*.json`：按机器 alias 保存服务状态，不是容器注册源。正在运行的服务建议先停止再改名。

注意当前 inventory 中至少有两台机器都叫 `xrs_vllm_main`：

- `80.5.17.111`
- `90.90.97.4`

因此需要明确目标宿主机 IP 和新容器名。比如：

```text
目标：90.90.97.4
旧名：xrs_vllm_main
新名：xrs_vllm_main1
```

然后我可以按安全流程帮你执行改名并同步 inventory。

> setup ssh怎么做的？

<details><summary>2 previous messages</summary>

> 我会用 `concise-code-explanation` 按代码调用链解释 SSH 初始化，重点拆成两层：宿主机 SSH（本地 → 宿主机）和容器 SSH（本地 → 容器）。我会只读查看相关实现，不执行 SSH 或修改远端。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content .agents/skills/concise-code-explanation/SKILL.md -Encoding UTF8; Get-Content .agents/skills/machine-management/SKILL.md -Encoding UTF8 | Select-Object -First 130; Write-Output '===== SSH call chain symbols ====='; rg -n "def (find_public_key|private_key_for_public_key|check_direct_ssh|build_bootstrap_host_key_command|run_with_askpass|bootstrap_host_key|check_host_key)|authorized_keys|sshd_vaws_config|PasswordAuthentication|PubkeyAuthentication|AllowUsers|ssh-keygen|ssh-keyscan|known_hosts|direct_container_ssh" .agents/skills/machine-management/scripts/machine_add.py .agents/skills/machine-management/scripts/_workflow_common.py .agents/skills/machine-management/scripts/manage_machine.py`
> - Ran `$f='.agents/skills/machine-management/scripts/manage_machine.py'; $ranges=@(@(600,710),@(2352,2425),@(1563,1730)); foreach($r in $ranges){Write-Output "===== ${f}:$($r[0])-$($r[1]) ====="; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object {$i++; if($i -ge $r[0] -and $i -le $r[1]) {'{0,4}: {1}' -f $i,$_}}}; $f2='.agents/skills/machine-management/scripts/_workflow_common.py'; $i=0; Get-Content $f2 -Encoding UTF8 | ForEach-Object {$i++; if($i -ge 410 -and $i -le 475) {'{0,4}: {1}' -f $i,$_}}`
> - Ran `$f='.agents/skills/machine-management/scripts/manage_machine.py'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object {$i++; if($i -ge 575 -and $i -le 605) {'{0,4}: {1}' -f $i,$_}}; $f2='.agents/skills/machine-management/scripts/machine_add.py'; $i=0; Get-Content $f2 -Encoding UTF8 | ForEach-Object {$i++; if($i -ge 162 -and $i -le 190) {'{0,4}: {1}' -f $i,$_}}`
>
> </details>
</details>

## 总述：SSH 初始化分两层

machine 注册时不是只配置一次 SSH，而是建立两条链路：

```text
本地机器
  |
  | SSH Key + 宿主机端口 22
  v
远端宿主机
  |
  | docker exec / 容器内配置
  v
容器专用 sshd
  |
  | SSH Key + 高位端口，例如 46000
  v
本地机器直连容器
```

宿主机 SSH 用于首次探测、Docker 操作和容器修复；容器 SSH 用于后续代码同步、服务启动、benchmark 等开发操作。

## 1. 宿主机 SSH：本地 → 宿主机

`machine_add.py` 首先寻找本地公钥，默认类似：

```text
~/.ssh/id_ed25519.pub
~/.ssh/id_ed25519
```

然后用 Key 测试宿主机 SSH：

```text
root@<host>:22
```

SSH 参数包含：

```text
BatchMode=yes
StrictHostKeyChecking=accept-new
IdentitiesOnly=yes
ConnectTimeout=10
```

如果 Key 登录成功，后续完全不使用密码。

如果 Key 登录失败，且用户提供了密码，脚本执行一次性 bootstrap：

1. 用密码登录宿主机；
2. 执行远端命令创建 `~/.ssh`；
3. 将本地公钥追加到宿主机的 `~/.ssh/authorized_keys`；
4. 再次使用 Key 登录验证。

远端写入是幂等的：

```bash
grep -qxF "$public_key" ~/.ssh/authorized_keys || \
printf '%s\n' "$public_key" >> ~/.ssh/authorized_keys
```

脚本没有使用 `ssh-copy-id`、`sshpass` 或 `expect`，而是通过临时 `SSH_ASKPASS` helper 将密码交给 SSH，完成后删除临时目录。

## 2. 容器 SSH：本地 → 容器

宿主机 SSH 建立后，bootstrap 脚本进入容器配置 SSH。

容器内首先写入本地公钥：

```text
/root/.ssh/authorized_keys
```

然后生成独立的 SSH 配置：

```text
/etc/ssh/sshd_vaws_config
```

关键配置是：

```text
Port 46000
PermitRootLogin prohibit-password
PubkeyAuthentication yes
PasswordAuthentication no
AuthorizedKeysFile .ssh/authorized_keys
PidFile /run/sshd_vaws.pid
```

也就是说：

- 容器 SSH 使用高位端口；
- 允许 root 使用 Key 登录；
- 禁止 root 密码登录；
- 不修改系统默认 `/etc/ssh/sshd_config`；
- 使用单独的 PID 文件和配置文件。

因为容器使用 `--network host`，所以容器 SSH 的访问地址实际是：

```text
<宿主机IP>:46000
```

例如 inventory 中：

```json
{
  "host": {
    "ip": "90.90.97.4"
  },
  "container": {
    "ssh_port": 46000
  }
}
```

最终连接就是：

```text
ssh -p 46000 root@90.90.97.4
```

代码通过 `container_target()` 从 inventory 读取宿主机 IP 和容器 SSH 端口。

## 3. 注册结束时的 SSH 验证

容器配置完成后，脚本会从本地直接连接容器，并执行：

```bash
printf ok
```

只有输出以 `ok` 结尾，才认为容器 SSH 正常。

最终 readiness 还会继续验证：

```text
宿主机 Key SSH
+
容器 Key SSH
+
torch / torch_npu NPU smoke
```

因此，单纯端口打开不等于 machine ready。

## 总结

SSH 注册链路是：

```text
本地公钥
  -> 一次性写入宿主机 authorized_keys
  -> 用宿主机 SSH 管理 Docker
  -> 将同一公钥写入容器 authorized_keys
  -> 启动容器专用高位端口 sshd
  -> 本地直连容器
  -> 执行 SSH + torch_npu readiness 检查
```

关键约束是：宿主机密码只用于首次安装公钥，容器从始至终不启用密码 SSH。

## 代码定位

1. [`machine_add.py`](./.agents/skills/machine-management/scripts/machine_add.py:162) — 查找本地 Key、检查宿主机 SSH、触发首次 bootstrap。
2. [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:422) — 宿主机公钥安装和密码边界。
3. [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:582) — 统一构造 SSH 命令和 Key 参数。
4. [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:645) — 生成写入宿主机 `authorized_keys` 的远端命令。
5. [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:410) — 根据 inventory 构造容器 SSH 目标。
6. [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:1580) — 写入容器公钥和配置专用 sshd。
7. [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:2360) — 执行最终 SSH 连通性检查。
8. [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:619) — 汇总宿主机 SSH、容器 SSH 和 NPU smoke 结果。

以上是基于当前工作区源码的静态分析，未实际执行 SSH。

> 配置97.44，已经有xrs_vllm_main了。我自己配置一下，把每一步的命令给我

<details><summary>1 previous message</summary>

> <details><summary>Ran 3 commands</summary>
>
> - Ran `$f='.agents/skills/machine-management/scripts/inventory.py'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object {$i++; if($i -ge 438 -and $i -le 570) {'{0,4}: {1}' -f $i,$_}}; $f2='.agents/skills/machine-management/scripts/manage_machine.py'; $i=0; Get-Content $f2 -Encoding UTF8 | ForEach-Object {$i++; if($i -ge 2748 -and $i -le 2875) {'{0,4}: {1}' -f $i,$_}}`
> - Ran `$f='.agents/skills/machine-management/scripts/manage_machine.py'; $i=0; Get-Content $f -Encoding UTF8 | ForEach-Object {$i++; if($i -ge 2853 -and $i -le 2945) {'{0,4}: {1}' -f $i,$_}}`
> - Ran `$inv=Get-Content .vaws-local/machine-inventory.json -Raw -Encoding UTF8 | ConvertFrom-Json; $inv.machines | Where-Object {$_.host.ip -eq '90.90.97.44'} | ConvertTo-Json -Depth 8`
>
> </details>
</details>

下面按 Windows PowerShell 给你一套“已有 `xrs_vllm_main` 容器、手动接入 `90.90.97.44`”的完整命令。

先说明：当前 inventory 中 `90.90.97.44` 记录的容器名还是 `vaws-agent27728`，不是 `xrs_vllm_main`。下面流程会以远端真实容器为准，并更新本地 inventory。

## 1. 设置变量

```powershell
$machineIp = "90.90.97.44"
$containerName = "xrs_vllm_main"
$containerPort = 46000
$machineUser = "agent27728"
$machineType = "A3"
$workdir = "/vllm-workspace"
$sshKey = "$env:USERPROFILE\.ssh\id_ed25519"
```

确认本地 SSH Key 存在：

```powershell
Test-Path "$sshKey.pub"
Test-Path $sshKey
```

如果不存在，再生成：

```powershell
ssh-keygen -t ed25519 -f $sshKey -C "vaws-machine"
```

## 2. 查看当前本地 profile 和 inventory

```powershell
py -3 .agents/scripts/workspace_profile.py summary
```

```powershell
py -3 .agents/skills/machine-management/scripts/inventory.py get $machineIp
```

## 3. 测试宿主机 SSH

```powershell
ssh `
  -i $sshKey `
  -p 22 `
  -o BatchMode=yes `
  -o StrictHostKeyChecking=accept-new `
  -o ConnectTimeout=10 `
  "root@$machineIp" `
  "printf ok"
```

如果输出：

```text
ok
```

说明宿主机 Key SSH 已经配置好，跳到第 5 步。

如果失败，需要用宿主机密码安装公钥：

```powershell
Read-Host "Host password" | py -3 .agents/skills/machine-management/scripts/manage_machine.py `
  bootstrap-host-key `
  --host $machineIp `
  --user root `
  --host-port 22 `
  --password-stdin
```

然后再次执行第 3 步确认。

## 4. 确认远端容器确实存在

```powershell
ssh `
  -i $sshKey `
  -p 22 `
  "root@$machineIp" `
  "docker ps -a --filter name=$containerName --format '{{.Names}} {{.Status}} {{.Image}}'"
```

读取远端容器实际镜像：

```powershell
$image = (ssh `
  -i $sshKey `
  -p 22 `
  "root@$machineIp" `
  "docker inspect -f '{{.Config.Image}}' $containerName").Trim()

$image
```

如果 `$image` 为空，说明远端不存在这个容器，先不要继续。

## 5. 探测宿主机环境

```powershell
py -3 .agents/skills/machine-management/scripts/manage_machine.py `
  probe-host `
  --host $machineIp `
  --user root `
  --host-port 22 `
  --image $image `
  --machine-type $machineType `
  --managed-prefix "xrs_"
```

重点确认输出中：

```text
docker.info_ok = true
prerequisites_ok = true
machine_type = A3
free_port = 46000
```

## 6. 配置已有容器的 SSH

这一步不会因为容器已有就重新创建；如果镜像匹配，会复用容器并配置专用 sshd。

```powershell
py -3 .agents/skills/machine-management/scripts/manage_machine.py `
  bootstrap-container `
  --host $machineIp `
  --user root `
  --host-port 22 `
  --container-name $containerName `
  --container-ssh-port $containerPort `
  --namespace $machineUser `
  --image $image `
  --machine-type $machineType `
  --workdir $workdir `
  --public-key-file "$sshKey.pub"
```

容器内会配置：

```text
/root/.ssh/authorized_keys
/etc/ssh/sshd_vaws_config
/etc/profile.d/vaws-ascend-env.sh
/etc/vaws/container-info.json
```

容器 SSH 端口是：

```text
90.90.97.44:46000
```

## 7. 测试直连容器 SSH

```powershell
ssh `
  -i $sshKey `
  -p $containerPort `
  -o BatchMode=yes `
  -o StrictHostKeyChecking=accept-new `
  -o ConnectTimeout=10 `
  "root@$machineIp" `
  "printf ok"
```

输出 `ok` 后，说明容器 SSH 已经成功。

也可以查看容器环境：

```powershell
ssh `
  -i $sshKey `
  -p $containerPort `
  "root@$machineIp" `
  "hostname; cat /etc/ssh/sshd_vaws_config"
```

## 8. 更新本地 inventory

不要手动删除整条 JSON 记录，使用 `upsert`，它会按 alias/IP 更新已有记录：

```powershell
py -3 .agents/skills/machine-management/scripts/inventory.py `
  upsert `
  --alias $machineIp `
  --namespace $machineUser `
  --host $machineIp `
  --host-port 22 `
  --host-user root `
  --host-machine-type $machineType `
  --container-name $containerName `
  --container-ssh-port $containerPort `
  --image $image `
  --workdir $workdir `
  --container-machine-type $machineType `
  --bootstrap-method ssh `
  --no-created-by-skill
```

这里的 `--no-created-by-skill` 表示容器原本是你自己创建的，不是注册脚本新建的，但仍然由 workspace 管理。

## 9. 最终完整验证

```powershell
py -3 .agents/skills/machine-management/scripts/machine_verify.py `
  --machine $machineIp
```

最终需要看到：

```json
{
  "status": "ready",
  "success": true
}
```

如果需要直接测试 NPU smoke：

```powershell
py -3 .agents/skills/machine-management/scripts/manage_machine.py `
  smoke `
  --host $machineIp `
  --container-ssh-port $containerPort
```

## 10. 检查 inventory 是否已更新

```powershell
py -3 .agents/skills/machine-management/scripts/inventory.py get $machineIp
```

最终应当是：

```json
{
  "alias": "90.90.97.44",
  "container": {
    "name": "xrs_vllm_main",
    "ssh_port": 46000
  }
}
```

不需要修改 `machine-profile.json`，也不需要修改 `.remote-dev/endpoints.local.json`。后续 parity、serving、benchmark 都会从 inventory 读取容器名和端口。

从源码看，宿主机公钥 bootstrap 位于 [`_workflow_common.py`](./.agents/skills/machine-management/scripts/_workflow_common.py:422)，容器 sshd 配置位于 [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:1580)，inventory 更新参数位于 [`inventory.py`](./.agents/skills/machine-management/scripts/inventory.py:512)。

> User context
>
> Attachments:
> - py -3 .agents/skills/machine-management/scripts/manage_machine.py ` bootstrap-c…: `C:/Users/x50063850/.codex/attachments/39554a1c-3b7a-485a-840a-0881f2d7477c/pasted-text.txt`

<details><summary>1 previous message</summary>

> System error
>
> Selected model is at capacity. Please try a different model.
</details>

> 继续看一下

> User context
>
> Attachments:
> - py -3 .agents/skills/machine-management/scripts/manage_machine.py ` bootstrap-c…: `C:/Users/x50063850/.codex/attachments/1959da3b-b6fe-4d22-9469-635e9d3f0d81/pasted-text.txt`

<details><summary>4 previous messages</summary>

> 我继续查看你贴出的命令输出，先读取附件内容，确认卡在哪一步、命令返回了什么，再判断是宿主机 SSH、容器 SSH、镜像参数还是 inventory 问题。不会重复执行可能有副作用的 bootstrap 命令。
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content 'C:\Users\x50063850\.codex\attachments\1959da3b-b6fe-4d22-9469-635e9d3f0d81\pasted-text.txt' -Encoding UTF8`
>
> </details>
>
> 输出已经定位：容器 bootstrap 本身成功，失败发生在最后的本地直连检查。原因是本机 `known_hosts` 中保存的 `[90.90.97.44]:46000` 旧容器 ECDSA host key 与当前容器新生成的 key 不一致；我先给你核验新指纹，再清理这一条旧记录。
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content .agents/skills/machine-management/SKILL.md -Encoding UTF8 | Select-Object -First 240; rg -n "def remove_known_host_entry|StrictHostKeyChecking=accept-new|def check_direct_ssh|direct_container_ssh" .agents/skills/machine-management/scripts/manage_machine.py .agents/skills/machine-management/scripts/_workflow_common.py`
>
> </details>
</details>

问题已经定位，不是容器 SSH 配置失败，而是本地 `known_hosts` 保存了旧容器的 host key。

远端 bootstrap 实际已经成功：

```text
reused-existing-container
configured-dedicated-sshd
wrote-host-metadata
configured-container-env
```

失败点是：

```text
Host key for [90.90.97.44]:46000 has changed
Offending ECDSA key in ...known_hosts:26
```

## 1. 先核验新 host key

不要直接删除，先从宿主机进入容器读取当前 key 指纹：

```powershell
$machineIp = "90.90.97.44"
$containerName = "xrs_vllm_main"
$sshKey = "$env:USERPROFILE\.ssh\id_ed25519"

ssh `
  -i $sshKey `
  -p 22 `
  "root@$machineIp" `
  "docker exec $containerName ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub -E sha256"
```

确认输出中包含：

```text
SHA256:T4RHc+SNjsVsIekJGd0ASdG985AP8HLwV+ITUE28UkE
```

如果指纹不一致，先不要删除 `known_hosts`，需要确认容器是否被替换或存在安全风险。

## 2. 删除本地旧容器 host key

确认指纹可信后，先查看旧记录：

```powershell
$knownHosts = "$env:USERPROFILE\.ssh\known_hosts"

ssh-keygen -F "[90.90.97.44]:46000" -f $knownHosts
```

删除这一台机器、这一端口的旧记录：

```powershell
ssh-keygen -R "[90.90.97.44]:46000" -f $knownHosts
```

该命令只删除 `[90.90.97.44]:46000`，不会影响宿主机的 `90.90.97.44:22`。

## 3. 重新测试容器 SSH

```powershell
ssh `
  -i $sshKey `
  -p 46000 `
  -o BatchMode=yes `
  -o StrictHostKeyChecking=accept-new `
  -o ConnectTimeout=10 `
  "root@$machineIp" `
  "printf ok"
```

预期输出：

```text
ok
```

## 4. 更新 inventory 中的容器名

你当前 bootstrap 使用的是：

```text
xrs_vllm_main
vllm-ascend:dev-26.1.0.cann9.1.0.day20260810-800I-A3-py311-Ubuntu24.04-lts-aarch64
```

执行：

```powershell
py -3 .agents/skills/machine-management/scripts/inventory.py `
  upsert `
  --alias 90.90.97.44 `
  --namespace agent27728 `
  --host 90.90.97.44 `
  --host-port 22 `
  --host-user root `
  --host-machine-type A3 `
  --container-name xrs_vllm_main `
  --container-ssh-port 46000 `
  --image "vllm-ascend:dev-26.1.0.cann9.1.0.day20260810-800I-A3-py311-Ubuntu24.04-lts-aarch64" `
  --workdir /vllm-workspace `
  --container-machine-type A3 `
  --bootstrap-method ssh `
  --no-created-by-skill
```

## 5. 最终验证

```powershell
py -3 .agents/skills/machine-management/scripts/machine_verify.py `
  --machine 90.90.97.44
```

预期：

```json
{
  "status": "ready",
  "success": true
}
```

这次不需要重新执行 `bootstrap-container`，因为容器配置已经完成；只需要清理本地旧 host key、更新 inventory，然后验证即可。

对应源码中，`StrictHostKeyChecking=accept-new` 位于 [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:582)，容器 SSH 后置检查位于 [`manage_machine.py`](./.agents/skills/machine-management/scripts/manage_machine.py:2360)。



## 代码同步流程
py -3 .agents/skills/remote-code-parity/scripts/parity_sync.py `
  --machine 90.90.97.44 `
  --print-derived-args

py -3 .agents/skills/remote-code-parity/scripts/install_consent.py `
  set-sync-mode `
  --repo-root . `
  --server-name 90.90.97.44 `
  --container-identity "xrs_vllm_main@/vllm-workspace" `
  --sync-mode local `
  --note "user requested current local workspace source sync" `
  --approved-by-user

当前 inventory 已解析为：

```text
机器：90.90.97.44
容器：xrs_vllm_main
端口：46000
运行目录：/vllm-workspace
```

## 只同步代码，不安装、不编译

先确认目标参数：

```powershell
py -3 .agents/skills/remote-code-parity/scripts/parity_sync.py `
  --machine 90.90.97.44 `
  --print-derived-args
```

当前本地 consent 里还没有 `xrs_vllm_main@/vllm-workspace`，先设置为使用本地代码：

```powershell
py -3 .agents/skills/remote-code-parity/scripts/install_consent.py `
  set-sync-mode `
  --repo-root . `
  --server-name 90.90.97.44 `
  --container-identity "xrs_vllm_main@/vllm-workspace" `
  --sync-mode local `
  --note "user requested current local workspace source sync" `
  --approved-by-user
```

执行代码同步：

```powershell
py -3 .agents/skills/remote-code-parity/scripts/parity_sync.py `
  --machine 90.90.97.44 `
  --apply-mode materialize
```

这个命令会同步当前本地工作树，包括未提交的非忽略文件，但不会执行 `pip install`、编译或重建。

## 只推送单个文件

例如只同步 `bench_serve.sh`：

```powershell
py -3 .agents/scripts/remote_artifact_push.py `
  --machine 90.90.97.44 `
  --local-path .\scripts\bench_serve.sh `
  --remote-path /vllm-workspace/scripts/bench_serve.sh
```

单文件推送适合临时脚本修改，不会更新完整 Git parity 状态。

## 同步并安装本地 vLLM

如果你希望远端运行本地 `vllm` 和 `vllm-ascend`，需要明确允许首次替换镜像内安装包：

```powershell
py -3 .agents/skills/remote-code-parity/scripts/install_consent.py `
  set-sync-mode `
  --repo-root . `
  --server-name 90.90.97.44 `
  --container-identity "xrs_vllm_main@/vllm-workspace" `
  --sync-mode local `
  --allow-first-install `
  --note "user approved local vllm and vllm-ascend editable install" `
  --approved-by-user
```

然后执行完整同步：

```powershell
py -3 .agents/skills/remote-code-parity/scripts/parity_sync.py `
  --machine 90.90.97.44 `
  --apply-mode install
```

当前需求建议使用 `materialize`；启动服务前如果需要真正使用本地 Python 包，再使用 `install`。

核心入口是 [`parity_sync.py`](D:/project/vllm-ascend-workspace/.agents/skills/remote-code-parity/scripts/parity_sync.py:169)，它会从 inventory 解析容器名、端口和运行目录。