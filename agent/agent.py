# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import json
import logging
import os
import pprint
import re
import subprocess
import sys
import threading
import time
import random
import tempfile
from abc import ABC, abstractmethod

from openai import AzureOpenAI, OpenAI

from common.common import num_tokens_from_messages
from config import cfg

# Agent-wide constants. Sourced from config/agent.yaml via OmegaConf. Kept as
# module-level names so existing callers (baseline/llmad.py, agent/image_agent.py,
# agent/repair_agent.py, agent/review_agent.py, agent/detection_agent.py,
# runtime/engine.py) continue to import them unchanged.
MAX_ITER = cfg.max_iter
TIMEOUT = cfg.timeout
TIMEOUT_FIRST_REVIEW = cfg.timeout_first_review
TIMEOUT_LLM = cfg.timeout_llm
TIMEOUT_PER_REVIEW = cfg.timeout_per_review
TIMEOUT_IMAGE = cfg.timeout_image
TIMEOUT_INFERENCE = cfg.timeout_inference
SELF_HOSTED_LLM_LIST = list(cfg.self_hosted_llm_list)
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL_ALIASES = {
    "gpt-4-mini": "gpt-4o-mini",
}
CODEX_MODEL_ALIASES = {
    "gpt-4-mini": "gpt-5.4-mini",
}

class LLM:
    def __init__(
        self,
        system_prompt: str = "You are an AI assistant that helps people find information.",
        engine: str = "gpt-4o",
        temperature: float = 0.1,
        past_message_num: int = sys.maxsize,
    ) -> None:
        self.provider = _resolve_llm_provider(engine)
        self.requested_engine = engine
        self.name = _resolve_engine_alias(engine, self.provider)
        self.engine = self.name
        self.system_prompt = system_prompt
        self.past_message_num = max(0, past_message_num)
        temperature = float(os.environ.get("ARGOS_LLM_TEMPERATURE", str(temperature)))
        # list of tuple of (input_token_count, output_token_count)
        self.input_output_token_count = []


        self._thread_local = threading.local()
        
        self.parameters = {
            "model": self.engine,
            "temperature": temperature,
            "max_tokens": 2000,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "n": 1,
        }

        if self.provider == "chatgpt-oauth":
            _assert_codex_oauth_available()
            self._openai = None
        elif self.provider == "openai":
            self._openai = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get(
                    "ARGOS_OPENAI_BASE_URL",
                    os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
                ),
            )
        elif self.provider == "self-hosted" or (
            self.provider == "auto" and self.engine in SELF_HOSTED_LLM_LIST
        ):
            self._openai = OpenAI(
                api_key=os.environ["OPENAI_AZURE_API_KEY"],
                base_url=os.environ["OPENAI_AZURE_ENDPOINT"],
            )
            # self._assert_openai_client_connect()
        else:
            self._openai = AzureOpenAI(
                api_key=os.environ["OPENAI_AZURE_API_KEY"],
                azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                api_version=os.environ["OPENAI_AZURE_API_VERSION"],
            )

        logging.info(f"Initialized LLM with provider={self.provider}")
        logging.info(f"Initialized LLM with the following parameters:")
        logging.info(pprint.pformat(self.parameters, width=120, compact=True))

    def _assert_openai_client_connect(self):
        try:
            _ = self._openai.chat.completions.create(
                messages= [
                    {
                        "role": "user",
                        "content": "Just say hello world back to me. Don't output anything else.",
                    }
                ],
                model=self.engine,
            )
        except Exception as e:
            logging.exception(e)
            raise Exception("OpenAI client connection failed.")
        
    def _init_messages(self):
        return [{"role": "system", "content": self.system_prompt}]
    
    def get_messages(self):
        if not hasattr(self._thread_local, "messages"):
            self._thread_local.messages = self._init_messages()
        return self._thread_local.messages

    def reset(self) -> None:
        self.update_messages(reset=True)

    def reset_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        messages = self.get_messages()
        messages[0]["content"] = self.system_prompt


    def update_messages(self, reset=False):
        if reset:
            self._thread_local.messages = self._init_messages()
        else:
            messages = self.get_messages()
            if self.past_message_num > 0:
                self._thread_local.messages = [messages[0]] + messages[1:][-self.past_message_num:]
            else:
                self._thread_local.messages = [messages[0]]
                
    def query(self, user_prompt: str) -> str:
        messages = self.get_messages()
        messages.append({"role": "user", "content": user_prompt})
        return self.send_messages()

    def query_with_image(self, user_prompt: str, image_path: str) -> str:

        assert self.engine == "gpt-4o", "Only gpt-4o engine supports image input."

        encoded_image = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
        content = [
            {
                "type": "text",
                "text": user_prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}",
                },
            },
        ]
        messages = self.get_messages()
        messages.append({"role": "user", "content": content})
        return self.send_messages()

    def send_messages(self):
        messages = self.get_messages()
        ans, timeout = "", 0.1
        input_token_count = num_tokens_from_messages(messages, model=self.engine)

        start_time = time.time()
        while not ans and time.time() - start_time < TIMEOUT_LLM:
            try:
                if self.provider == "chatgpt-oauth":
                    ans = self._query_codex_cli(messages)
                else:
                    response = self._openai.chat.completions.create(
                        messages=messages, **self.parameters,
                    )
                    if isinstance(response, list):
                        response = response[0]
                        if response.error:
                            raise Exception(response.error)
                    ans = response.choices[0].message.content
            except Exception as e:
                logging.exception(e)
                if _is_non_retryable_llm_error(e):
                    raise

            if not ans:
                wait_time = min(timeout * 2, TIMEOUT_LLM)
                wait_time += random.uniform(0, 1)
                logging.info(f"Will retry after {wait_time:.2f} seconds ...")
                time.sleep(wait_time)
                timeout = wait_time

        if not ans:
            raise TimeoutError(f"Timeout after {TIMEOUT_LLM} seconds.")

        elapsed_time = time.time() - start_time
        logging.info(f"[LLM] Query {self.name} finished after {elapsed_time:.2f} seconds")
        logging.info(ans)

        messages.append({"role": "assistant", "content": ans})
        output_token_count = num_tokens_from_messages([messages[-1]], model=self.engine)
        
        logging.info(f"[LLM] Input tokens: {input_token_count}, Output tokens: {output_token_count}")

        with self._token_count_lock:
            self.input_output_token_count.append((input_token_count, output_token_count))

        self.update_messages()
        return ans

    def _query_codex_cli(self, messages) -> str:
        prompt = _messages_to_codex_prompt(messages)
        codex_cli = os.environ.get("ARGOS_CODEX_CLI", "codex")
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, suffix=".txt"
        ) as output_file:
            output_path = output_file.name
        try:
            command = [
                codex_cli,
                "exec",
                "--model",
                self.engine,
                "--sandbox",
                "read-only",
                "--output-last-message",
                output_path,
                "-",
            ]
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_LLM,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Codex OAuth LLM call failed: "
                    + (completed.stderr or completed.stdout or "").strip()
                )
            with open(output_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        finally:
            try:
                os.remove(output_path)
            except OSError:
                pass

    def get_token_count(self):
        with self._token_count_lock:
            total_input_token_count = sum(x[0] for x in self.input_output_token_count)
            total_output_token_count = sum(x[1] for x in self.input_output_token_count)
            return (
                list(self.input_output_token_count),
                total_input_token_count,
                total_output_token_count,
            )


def _resolve_llm_provider(engine: str) -> str:
    provider = os.environ.get("ARGOS_LLM_PROVIDER", "auto").strip().lower()
    valid_providers = {"auto", "azure", "openai", "chatgpt-oauth", "self-hosted"}
    if provider not in valid_providers:
        raise ValueError(
            f"Invalid ARGOS_LLM_PROVIDER={provider}. "
            f"Expected one of {sorted(valid_providers)}."
        )
    if provider == "azure":
        return "azure"
    return provider


def _resolve_engine_alias(engine: str, provider: str) -> str:
    if provider == "chatgpt-oauth":
        return CODEX_MODEL_ALIASES.get(engine, engine)
    return OPENAI_MODEL_ALIASES.get(engine, engine)


def _messages_to_codex_prompt(messages) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        parts.append(f"##### {role}\n{content}")
    return "\n\n".join(parts)


def _assert_codex_oauth_available() -> None:
    # Do not read token contents here. Codex CLI handles ChatGPT OAuth via CODEX_HOME.
    auth_path = os.environ.get(
        "ARGOS_CODEX_AUTH_PATH",
        os.path.join(os.path.expanduser("~"), ".codex", "auth.json"),
    )
    if not os.path.exists(auth_path):
        raise RuntimeError(
            "chatgpt-oauth provider requires a Codex/ChatGPT login at ~/.codex/auth.json."
        )


def _load_oauth_access_token() -> str:
    token = os.environ.get("ARGOS_OPENAI_OAUTH_TOKEN") or os.environ.get(
        "OPENAI_OAUTH_TOKEN"
    )
    if token:
        return token

    auth_path = os.environ.get(
        "ARGOS_CODEX_AUTH_PATH",
        os.path.join(os.path.expanduser("~"), ".codex", "auth.json"),
    )
    with open(auth_path, "r", encoding="utf-8") as f:
        auth = json.load(f)

    token = auth.get("tokens", {}).get("access_token")
    if not token:
        raise RuntimeError(
            "ChatGPT OAuth provider requires ARGOS_OPENAI_OAUTH_TOKEN, "
            "OPENAI_OAUTH_TOKEN, or tokens.access_token in ~/.codex/auth.json."
        )
    return token


def _is_non_retryable_llm_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in {400, 401, 403, 404, 429}:
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "invalid api key",
            "incorrect api key",
            "unauthorized",
            "permission denied",
            "model_not_found",
            "does not exist",
            "insufficient_quota",
            "exceeded your current quota",
            "stream disconnected before completion",
            "error sending request for url",
            "failed to connect to websocket",
            "os error 10013",
        )
    )


class Agent(ABC):
    LLM: LLM
    name: str

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def name(self):
        return self.name

    def extract_code(
        self, text: str, lang: str = "python", function_name: str = "def inference"
    ) -> str:
        start, end = f"```{lang}", "```"
        # use regex to extract the code block that start with start and end with end
        pattern = re.compile(f"{start}.*{end}", re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            if function_name in match:
                return match.replace(start, "").replace(end, "").strip()
        raise ValueError(
            f"Cannot find code block with function name {function_name} in the text."
        )

    def save_rule(self, rule: str, save_path: str) -> None:
        with open(save_path, "w") as f:
            f.write(rule)
        logging.info(f"Write rule to {save_path}")

    def set_start_time(self) -> None:
        self.start_time = time.time()

    def get_elapsed_time(self) -> float:
        assert hasattr(self, "start_time"), "Please set start time first."
        return time.time() - self.start_time

    def get_token_count(self):
        return self.LLM.get_token_count()
