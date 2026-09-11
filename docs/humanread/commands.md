## profiling command
PID=3552871
CID=$(grep -oE '[0-9a-f]{64}' /proc/$PID/cgroup | head -1)
docker ps --no-trunc | grep "$CID"

## 回退单个commit单个文件修改
SP pr = 92616f3a64e7218655c2d009aa8ecf855091e54d