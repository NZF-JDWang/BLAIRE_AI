import os

from app.core.config import Settings
from app.services.tool_policy import ToolPolicy, ToolPolicyError
from app.tools.base import ToolSpec


def test_network_tool_requires_allowlisted_host() -> None:
    os.environ["DATABASE_URL"] = "postgresql+psycopg://user:pass@localhost:5432/db"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["MCP_OBSIDIAN_URL"] = "http://localhost:3000"
    os.environ["MCP_HA_URL"] = "http://localhost:3001"
    os.environ["MODEL_GENERAL_DEFAULT"] = "qwen2.5:7b-instruct"
    os.environ["MODEL_VISION_DEFAULT"] = "qwen2.5vl:7b"
    os.environ["MODEL_EMBEDDING_DEFAULT"] = "nomic-embed-text:v1.5"
    os.environ["ALLOWED_NETWORK_HOSTS"] = "host-a,host-b"
    os.environ["ALLOWED_NETWORK_TOOLS"] = "network_probe"

    settings = Settings()
    policy = ToolPolicy(settings)
    spec = ToolSpec(
        name="network_probe",
        action_class="network_sensitive",
        description="test",
        requires_target_host=True,
    )

    policy.validate_network_tool(spec, "host-a")

    try:
        policy.validate_network_tool(spec, "host-c")
        assert False, "Expected ToolPolicyError"
    except ToolPolicyError:
        assert True

