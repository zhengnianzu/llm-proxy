import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils import backup_routes, backup_store, obs_utils
from utils.backup_routes import _parse_raw_index, _parse_session_index, _safe_relative_file


class SelectiveRawDownloadTest(unittest.TestCase):
    def test_resolver_and_single_object_command(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "demo-bucket"
            binary = root / "obsutil"
            original_config = b"endpoint=https://example.invalid\nak=demo\nsk=secret\n"
            config.write_bytes(original_config)
            binary.write_text("binary", encoding="utf-8")
            target = root / "out" / "file.json"
            runtime_dir = root / "runtime-config"
            observed = {}

            def fake_run(command, **_kwargs):
                config_arg = next(
                    item for item in command if item.startswith("-config=")
                )
                runtime_config = Path(config_arg.split("=", 1)[1])
                self.assertNotEqual(runtime_config.resolve(), config.resolve())
                self.assertEqual(runtime_config.read_bytes(), original_config)
                if os.name != "nt":
                    self.assertEqual(
                        stat.S_IMODE(runtime_config.stat().st_mode), 0o600
                    )
                observed["runtime_config"] = runtime_config
                runtime_config.write_text("obsutil-mutated-copy", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            env = {
                "OBSUTIL_CONFIG_DIR": temp,
                "OBSUTIL_RUNTIME_CONFIG_DIR": str(runtime_dir),
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(
                    obs_utils.resolve_obsutil_config("obs://demo-bucket/a.json"),
                    str(config.resolve()),
                )
                with patch.object(obs_utils, "OBSUTIL_BIN", str(binary)):
                    with patch.object(
                        obs_utils.subprocess,
                        "run",
                        side_effect=fake_run,
                    ) as run:
                        ok, _ = obs_utils.download_obs_object(
                            "obs://demo-bucket/a.json", str(target)
                        )
            self.assertTrue(ok)
            self.assertEqual(config.read_bytes(), original_config)
            self.assertFalse(observed["runtime_config"].exists())
            command = run.call_args.args[0]
            self.assertNotIn("-r", command)
            self.assertNotIn(f"-config={config.resolve()}", command)
            self.assertTrue(
                any(item.startswith("-config=") for item in command)
            )

    def test_session_index_prefers_complete_successful_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "session_index.jsonl"
            rows = [
                {
                    "session": "s1",
                    "q1": "你好",
                    "last_ts": "2026-07-31_12-00-00",
                    "models": ["m1"],
                    "msg_count": 8,
                    "trace_list": [
                        {
                            "filename": "good-req.json",
                            "msg_count": 8,
                            "success": True,
                        },
                        {
                            "filename": "failed-req.json",
                            "msg_count": 20,
                            "success": False,
                        },
                    ],
                },
                {"_meta": True, "version": 2},
            ]
            index.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            items = _parse_session_index(index, "obs://demo-bucket/raw/a/")
            self.assertEqual(len(items), 1)
            self.assertEqual(
                items[0]["_files"], ["good-req.json", "good-res.json"]
            )
            self.assertEqual(items[0]["files_count"], 2)

    def test_config_copy_is_removed_after_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "demo-bucket"
            binary = root / "obsutil"
            runtime_dir = root / "runtime-config"
            original_config = b"endpoint=https://example.invalid\nak=demo\nsk=secret\n"
            config.write_bytes(original_config)
            binary.write_text("binary", encoding="utf-8")

            env = {
                "OBSUTIL_CONFIG_DIR": temp,
                "OBSUTIL_RUNTIME_CONFIG_DIR": str(runtime_dir),
            }
            with patch.dict(os.environ, env, clear=False):
                with patch.object(obs_utils, "OBSUTIL_BIN", str(binary)):
                    with patch.object(
                        obs_utils.subprocess,
                        "run",
                        side_effect=obs_utils.subprocess.TimeoutExpired(
                            cmd="obsutil", timeout=1
                        ),
                    ):
                        ok, message = obs_utils.download_obs_object(
                            "obs://demo-bucket/a.json",
                            str(root / "out" / "a.json"),
                            timeout=1,
                        )

            self.assertFalse(ok)
            self.assertIn("超时", message)
            self.assertEqual(config.read_bytes(), original_config)
            self.assertEqual(list(runtime_dir.iterdir()), [])

    def test_obsutil_ls_uses_and_removes_private_config_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "demo-bucket"
            binary = root / "obsutil"
            runtime_dir = root / "runtime-config"
            original_config = b"endpoint=https://example.invalid\nak=demo\nsk=secret\n"
            config.write_bytes(original_config)
            binary.write_text("binary", encoding="utf-8")
            observed = {}

            def fake_run(command, **_kwargs):
                config_arg = next(
                    item for item in command if item.startswith("-config=")
                )
                runtime_config = Path(config_arg.split("=", 1)[1])
                observed["runtime_config"] = runtime_config
                self.assertEqual(runtime_config.read_bytes(), original_config)
                runtime_config.write_text("obsutil-mutated-copy", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.dict(
                os.environ,
                {"OBSUTIL_RUNTIME_CONFIG_DIR": str(runtime_dir)},
                clear=False,
            ):
                with patch.object(obs_utils, "OBSUTIL_BIN", str(binary)):
                    with patch.object(
                        obs_utils.subprocess, "run", side_effect=fake_run
                    ):
                        result = obs_utils.obsutil_ls(
                            "obs://demo-bucket/a/",
                            config_path=str(config),
                        )

            self.assertEqual(result, [])
            self.assertEqual(config.read_bytes(), original_config)
            self.assertFalse(observed["runtime_config"].exists())

    def test_raw_index_fallback_prefers_success_and_hides_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "index.jsonl"
            rows = [
                {
                    "ts": "2026-07-31_12-00-00",
                    "req_file": "good-req.json",
                    "api_key": "secret-key",
                    "q1_hash": "same-session",
                    "q1_preview": "你好",
                    "model": "m1",
                    "msg_count": 8,
                    "success": True,
                },
                {
                    "ts": "2026-07-31_12-01-00",
                    "req_file": "failed-req.json",
                    "api_key": "secret-key",
                    "q1_hash": "same-session",
                    "q1_preview": "你好",
                    "model": "m1",
                    "msg_count": 20,
                    "success": False,
                },
            ]
            index.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )
            items = _parse_raw_index(index, "obs://demo-bucket/raw/a/")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["_files"], ["good-req.json", "good-res.json"])
            self.assertNotIn("api_key", items[0])
            self.assertNotIn("secret-key", json.dumps(items[0], ensure_ascii=False))

    def test_empty_session_index_falls_back_to_raw_index(self):
        obs_path = "obs://demo-bucket/raw/env/empty-session-index/"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session_index = root / "session_index.jsonl"
            raw_index = root / "index.jsonl"
            session_index.write_text(
                json.dumps({"_meta": True, "version": 2}),
                encoding="utf-8",
            )
            raw_index.write_text(
                json.dumps(
                    {
                        "ts": "2026-07-31_12-00-00",
                        "req_file": "fallback-req.json",
                        "q1_hash": "fallback-session",
                        "q1_preview": "回退会话",
                        "model": "m1",
                        "success": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def fake_download(_obs_path, filename, _config_path):
                return (
                    session_index if filename == "session_index.jsonl" else raw_index,
                    "",
                )

            backup_routes._RAW_CATALOG_CACHE.pop(obs_path, None)
            with patch.object(
                backup_routes,
                "resolve_obsutil_config",
                return_value=str(root / "demo-bucket"),
            ):
                with patch.object(
                    backup_routes,
                    "_download_catalog_file",
                    side_effect=fake_download,
                ):
                    items, source, warning = backup_routes._load_raw_catalog(obs_path)
            backup_routes._RAW_CATALOG_CACHE.pop(obs_path, None)

            self.assertEqual(source, "index.jsonl")
            self.assertEqual(warning, "")
            self.assertEqual(len(items), 1)
            self.assertEqual(
                items[0]["_files"],
                ["fallback-req.json", "fallback-res.json"],
            )

    def test_cached_catalog_skips_empty_session_index(self):
        obs_path = "obs://demo-bucket/raw/env/cached-fallback/"
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"OBS_DOWNLOAD_ROOT": temp}, clear=False):
                session_cache = backup_routes._catalog_cache_file(
                    obs_path, "session_index.jsonl"
                )
                raw_cache = backup_routes._catalog_cache_file(obs_path, "index.jsonl")
                session_cache.write_text(
                    json.dumps({"_meta": True, "version": 2}),
                    encoding="utf-8",
                )
                raw_cache.write_text(
                    json.dumps(
                        {
                            "ts": "2026-07-31_12-00-00",
                            "req_file": "cached-req.json",
                            "q1_hash": "cached-fallback",
                            "q1_preview": "缓存回退会话",
                            "model": "m1",
                            "success": True,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                items, source = backup_routes._load_cached_raw_catalog(obs_path)

            self.assertEqual(source, "index.jsonl")
            self.assertIsNotNone(items)
            self.assertEqual(len(items), 1)
            self.assertEqual(
                items[0]["_files"],
                ["cached-req.json", "cached-res.json"],
            )

    def test_catalog_uses_last_good_disk_cache_when_obs_auth_fails(self):
        obs_path = "obs://demo-bucket/raw/env/mtime/"
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"OBS_DOWNLOAD_ROOT": temp}, clear=False):
                cache_file = backup_routes._catalog_cache_file(
                    obs_path, "index.jsonl"
                )
                cache_file.write_text(
                    json.dumps(
                        {
                            "ts": "2026-07-31_12-00-00",
                            "req_file": "good-req.json",
                            "q1_hash": "cached-session",
                            "q1_preview": "缓存会话",
                            "model": "m1",
                            "success": True,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                backup_routes._RAW_CATALOG_CACHE.pop(obs_path, None)
                with patch.object(
                    backup_routes,
                    "resolve_obsutil_config",
                    return_value=str(Path(temp) / "demo-bucket"),
                ):
                    with patch.object(
                        backup_routes,
                        "_download_catalog_file",
                        return_value=(None, "Status [403] InvalidAccessKeyId"),
                    ):
                        items, source, warning = backup_routes._load_raw_catalog(
                            obs_path
                        )
                self.assertEqual(len(items), 1)
                self.assertEqual(source, "index.jsonl")
                self.assertIn("凭据", warning)
                self.assertNotIn(obs_path, backup_routes._RAW_CATALOG_CACHE)

    def test_catalog_reports_auth_error_instead_of_missing_index(self):
        obs_path = "obs://demo-bucket/raw/env/no-cache/"
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"OBS_DOWNLOAD_ROOT": temp}, clear=False):
                backup_routes._RAW_CATALOG_CACHE.pop(obs_path, None)
                with patch.object(
                    backup_routes,
                    "resolve_obsutil_config",
                    return_value=str(Path(temp) / "demo-bucket"),
                ):
                    with patch.object(
                        backup_routes,
                        "_download_catalog_file",
                        return_value=(None, "Status [403] InvalidAccessKeyId"),
                    ):
                        with self.assertRaises(
                            backup_routes._RawCatalogError
                        ) as raised:
                            backup_routes._load_raw_catalog(obs_path)
                self.assertIn("凭据", str(raised.exception))
                self.assertNotIn("未找到", str(raised.exception))

    def test_relative_file_rejects_parent_traversal(self):
        self.assertEqual(_safe_relative_file("../secret.json"), "")
        self.assertEqual(_safe_relative_file("nested/../../secret.json"), "")
        self.assertEqual(_safe_relative_file("/tmp/trace-req.json"), "trace-req.json")

    def test_relative_obs_config_mapping_cannot_escape_config_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            env = {
                "OBSUTIL_CONFIG_DIR": temp,
                "OBSUTIL_CONFIG_MAP": json.dumps({"demo-bucket": "../secret"}),
            }
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaises(ValueError):
                    obs_utils.resolve_obsutil_config(
                        "obs://demo-bucket/a.json", require_exists=False
                    )

    def test_download_job_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            try:
                backup_store.init_db(temp)
                backup_store.create_raw_download_job(
                    "a" * 32,
                    "env/mtime",
                    "obs://demo-bucket/raw/env/mtime/",
                    str(Path(temp) / "downloads" / ("a" * 32)),
                    [{"session_id": "b" * 24, "relative_path": "x.json"}],
                )
                job = backup_store.get_raw_download_job("a" * 32)
                self.assertEqual(job["status"], "queued")
                self.assertEqual(job["total_files"], 1)
                backup_store.update_raw_download_job(
                    "a" * 32, status="completed", downloaded_files=1
                )
                job = backup_store.get_raw_download_job("a" * 32)
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["downloaded_files"], 1)
            finally:
                backup_store._conn.close()
                backup_store._conn = None


if __name__ == "__main__":
    unittest.main()
