import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.agent import _load_oauth_access_token, _resolve_llm_provider
from agent.agent import _resolve_engine_alias


class OAuthProviderTests(unittest.TestCase):
    def test_loads_oauth_token_from_env(self):
        with patch.dict(os.environ, {"ARGOS_OPENAI_OAUTH_TOKEN": "env-token"}, clear=False):
            self.assertEqual(_load_oauth_access_token(), "env-token")

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
                self.assertEqual(_load_oauth_access_token(), "file-token")

    def test_resolves_chatgpt_oauth_provider(self):
        with patch.dict(os.environ, {"ARGOS_LLM_PROVIDER": "chatgpt-oauth"}, clear=False):
            self.assertEqual(_resolve_llm_provider("gpt-4o"), "chatgpt-oauth")

    def test_maps_gpt4_mini_for_codex_oauth(self):
        self.assertEqual(
            _resolve_engine_alias("gpt-4-mini", "chatgpt-oauth"), "gpt-5.4-mini"
        )
        self.assertEqual(_resolve_engine_alias("gpt-4-mini", "openai"), "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
