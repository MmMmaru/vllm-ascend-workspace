---
name: e2e-test
description: 指导在 vllm-ascend 中准备并运行 E2E 测试；适用于需要临时替换模型、同步代码到远程 NPU 服务器、执行 pytest、查看日志并在提交前清理测试改动的场景。
---

# E2E Test

在仓库根目录执行。E2E 测试通常需要真实 NPU、匹配的依赖和可用模型权重。

## 流程

1. 确认测试文件、卡数、启动参数和模型路径。优先使用测试已有的模型配置。
2. 如果脚本把模型写死，只做最小的临时修改，使其读取：

   ```python
   os.environ.get("TEST_MODEL", "Qwen/Qwen3-30B-A3B")
   ```

   同时补齐 `import os`，不要修改无关逻辑。
3. 检查远程资源并选择空闲机器：

   ```bash
   ./remote machines
   ```

   按需读取 `.remote/state/docs/<alias>.facts.json`；缺失或过期时先运行
   `./remote verify <alias>`。所有远程操作都使用 `./remote`，不要直接使用 ssh。
4. 同步当前代码：

   ```bash
   ./remote sync <alias>
   ```

5. 在服务器上运行目标测试。长任务使用后台 job，并声明实际占用的卡：

   ```bash
   ./remote run <alias> --background --task "E2E <test-name>" \
     --cards 0,1 --env TEST_MODEL=Qwen/Qwen3-30B-A3B \
     --cmd "set -o pipefail; mkdir -p .log; \
       pytest -sv <test-path> 2>&1 | tee .log/e2e_<name>.log" \
     --timeout 3600 --logs full
   ```

   根据测试实际需要替换模型、卡号、测试路径和超时时间。记录返回的 `job_id`，用
   `./remote logs <job-id> --tail 200` 查看进度，最后用
   `./remote jobs --machine <alias>`
   确认任务结束并释放占用。
6. 验收退出码、pytest 结果和日志中的真实功能输出；记录模型、卡数、命令和日志位置。

## 提交前清理

如果模型读取逻辑只是为了本地验证而临时加入，测试完成后恢复测试脚本并删除临时日志、输出
和调试代码。提交前检查：

```bash
git diff --check
git status --short
```

正式需要保留的测试改动必须按仓库约定提交，不能把临时 E2E 修改混入提交。
