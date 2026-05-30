import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_agent_module():
    fake_openai = types.SimpleNamespace(AzureOpenAI=object, OpenAI=object)
    fake_config = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            max_iter=1,
            timeout=1,
            timeout_first_review=1,
            timeout_llm=1,
            timeout_per_review=1,
            timeout_image=1,
            timeout_inference=1,
            self_hosted_llm_list=[],
        )
    )
    fake_tiktoken = types.SimpleNamespace(
        encoding_for_model=lambda model: types.SimpleNamespace(encode=lambda text: []),
        get_encoding=lambda name: types.SimpleNamespace(encode=lambda text: []),
    )
    with patch.dict("sys.modules", {"openai": fake_openai, "config": fake_config, "tiktoken": fake_tiktoken}):
        import importlib

        return importlib.import_module("agent.agent")




AGENT_MODULE = _load_agent_module()


class OAuthProviderTests(unittest.TestCase):
    def test_loads_oauth_token_from_env(self):
        with patch.dict(os.environ, {"ARGOS_OPENAI_OAUTH_TOKEN": "env-token"}, clear=False):
            self.assertEqual(AGENT_MODULE._load_oauth_access_token(), "env-token")

    def test_loads_oauth_token_from_codex_auth_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            auth_path.write_text(
                json.dumps({"tokens": {"access_token": "file-token"}}),
                encoding="utf-8",
            )
            env = {
                "ARGOS_CODEX_AUTH_PATH": str(auth_path),
                "ARGOS_OPENAI_OAUTH_TOKEN": "",
                "OPENAI_OAUTH_TOKEN": "",
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(AGENT_MODULE._load_oauth_access_token(), "file-token")

    def test_resolves_chatgpt_oauth_provider(self):
        with patch.dict(os.environ, {"ARGOS_LLM_PROVIDER": "chatgpt-oauth"}, clear=False):
            self.assertEqual(AGENT_MODULE._resolve_llm_provider("gpt-4o"), "chatgpt-oauth")

    def test_maps_gpt4_mini_for_codex_oauth(self):
        self.assertEqual(
            AGENT_MODULE._resolve_engine_alias("gpt-4-mini", "chatgpt-oauth"), "gpt-5.4-mini"
        )
        self.assertEqual(AGENT_MODULE._resolve_engine_alias("gpt-4-mini", "openai"), "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
