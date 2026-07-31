"""
utils/atomic_write.py — 跨文件系统的「原子替换」封装

os.replace(tmp, dst) 在普通本地盘（ext4/xfs）上是原子的，是写临时文件再改名的
标准落盘套路。但部分 NFS 挂载（如华为云 SFS Turbo）**不支持 rename 操作**，
os.replace 会抛 PermissionError [Errno 1] Operation not permitted——即便进程是
root、目录 777、目标属主一致也一样，纯粹是服务端不实现 RENAME RPC。

safe_replace() 优先走 os.replace 保留原子性；仅在 rename 被拒时降级为
「覆盖写目标 + 删临时文件」。降级路径牺牲原子性（写一半崩溃会留残缺文件），
换取在这类存储上的可用性——调用方多为索引/缓存类，重建即可。
"""

import os
import shutil


def safe_replace(tmp_path: str, dst_path: str) -> None:
    """把 tmp_path 落成 dst_path。正常盘走原子 rename；不支持 rename 的
    NFS 挂载降级为覆盖写 + 删临时文件。"""
    tmp_path = str(tmp_path)
    dst_path = str(dst_path)
    try:
        os.replace(tmp_path, dst_path)
    except OSError:
        # 某些 NFS 挂载不支持 rename（EPERM）；降级为覆盖写，牺牲原子性换可用性。
        shutil.copyfile(tmp_path, dst_path)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
