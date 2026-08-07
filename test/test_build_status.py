"""
test_build_status.py — 测试构建状态显示逻辑
"""
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logs_routes import _describe


def test_build_status_with_active_backfill():
    """测试有活跃回填任务时的构建状态"""

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 模拟源数据
        src = {
            "root_id": "test_root",
            "root_path": tmpdir,
            "name": "test_source",
            "format": "newapi",
            "templates": []
        }

        # Mock logdir_store 模块
        with patch('utils.logs_routes.get_root_id') as mock_get_root_id, \
             patch('utils.logdir_store.has_any') as mock_has_any, \
             patch('utils.logdir_store.count_summary') as mock_count_summary, \
             patch('utils.logdir_store.has_active_backfill') as mock_has_active, \
             patch('utils.logdir_store.get_active_backfill_request') as mock_get_request:

            mock_get_root_id.return_value = "test_root"
            mock_has_any.return_value = True
            mock_count_summary.return_value = {
                "total": 100,
                "built": 45,
                "pending": 55,
                "building": 0,
                "error": 0
            }

            # 场景1: 有活跃任务且状态为 running
            mock_has_active.return_value = True
            mock_get_request.return_value = {"status": "running"}

            result = _describe(src, active=True)

            assert result["build_status"] == "building", \
                f"预期 build_status='building', 实际得到 '{result['build_status']}'"
            assert result["leaf_count"] == 100
            assert result["built_count"] == 45
            print("✓ 场景1通过: 活跃任务状态为 running 时显示 'building'")

            # 场景2: 有活跃任务但状态为 pending
            mock_get_request.return_value = {"status": "pending"}

            result = _describe(src, active=True)

            assert result["build_status"] == "queued", \
                f"预期 build_status='queued', 实际得到 '{result['build_status']}'"
            print("✓ 场景2通过: 活跃任务状态为 pending 时显示 'queued'")

            # 场景3: 没有活跃任务
            mock_has_active.return_value = False

            result = _describe(src, active=True)

            assert result["build_status"] == "none", \
                f"预期 build_status='none', 实际得到 '{result['build_status']}'"
            print("✓ 场景3通过: 无活跃任务时显示 'none'")


def test_build_status_normpath_call():
    """测试 has_active_backfill 使用正确的标准化路径"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # 使用带有冗余路径的目录
        path_with_dots = os.path.join(tmpdir, "subdir", "..", "subdir")
        os.makedirs(os.path.join(tmpdir, "subdir"), exist_ok=True)

        src = {
            "root_id": "test_root",
            "root_path": path_with_dots,
            "name": "test_source",
            "format": "newapi",
            "templates": []
        }

        with patch('utils.logs_routes.get_root_id') as mock_get_root_id, \
             patch('utils.logdir_store.has_any') as mock_has_any, \
             patch('utils.logdir_store.count_summary') as mock_count_summary, \
             patch('utils.logdir_store.has_active_backfill') as mock_has_active, \
             patch('utils.logdir_store.get_active_backfill_request') as mock_get_request:

            mock_get_root_id.return_value = "test_root"
            mock_has_any.return_value = True
            mock_count_summary.return_value = {"total": 10, "built": 5}
            mock_has_active.return_value = True
            mock_get_request.return_value = {"status": "running"}

            result = _describe(src, active=True)

            # 验证调用时使用了标准化路径
            called_path = mock_has_active.call_args[0][0]
            assert called_path == os.path.normpath(path_with_dots), \
                f"预期调用标准化路径 '{os.path.normpath(path_with_dots)}', 实际调用 '{called_path}'"

            # 验证两次调用使用相同的标准化路径
            assert mock_has_active.call_args[0][0] == mock_get_request.call_args[0][0], \
                "has_active_backfill 和 get_active_backfill_request 应该使用相同的标准化路径"

            print("✓ 路径标准化测试通过: 使用 os.path.normpath() 标准化路径")


if __name__ == "__main__":
    print("开始测试构建状态逻辑...")
    print()

    try:
        test_build_status_with_active_backfill()
        print()
        test_build_status_normpath_call()
        print()
        print("=" * 50)
        print("✅ 所有测试通过!")
        print("=" * 50)
    except AssertionError as e:
        print()
        print("=" * 50)
        print(f"❌ 测试失败: {e}")
        print("=" * 50)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        sys.exit(1)
