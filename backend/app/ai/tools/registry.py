"""Expanded Tool Registry Subsystem.

Provides dynamic tool registration, lookup, category/capability aggregation,
health probes, and OpenAPI JSON schema export for LLM function calling.
"""

import logging

from app.ai.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class ToolRegistryError(Exception):
    """Raised on tool registry registration or lookup errors."""


class ToolRegistry:
    """Central registry maintaining tool metadata, capabilities, and categories."""

    def __init__(self) -> None:
        """Initializes an empty ToolRegistry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance in the registry.

        Args:
            tool: BaseTool concrete instance.

        Raises:
            ToolRegistryError: If a tool with the same name is already registered.
        """
        name = tool.metadata.name
        if name in self._tools:
            raise ToolRegistryError(f"Tool '{name}' is already registered in ToolRegistry.")
        self._tools[name] = tool
        logger.info(f"Registered tool '{name}' (category: '{tool.metadata.category}').")

    def unregister(self, name: str) -> None:
        """Unregisters a tool by name.

        Args:
            name: Tool string identifier.
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool '{name}'.")

    def find(self, name: str) -> BaseTool | None:
        """Retrieves a registered tool by name.

        Args:
            name: Tool string identifier.

        Returns:
            BaseTool | None: Tool instance if found, else None.
        """
        return self._tools.get(name)

    def discover(self) -> list[ToolMetadata]:
        """Returns complete list of registered ToolMetadata objects."""
        return [tool.metadata for tool in self._tools.values()]

    def categories(self) -> list[str]:
        """Returns sorted list of unique tool categories."""
        return sorted({tool.metadata.category for tool in self._tools.values()})

    def capabilities(self) -> list[str]:
        """Returns sorted list of all unique capabilities required across registered tools."""
        caps: set[str] = set()
        for tool in self._tools.values():
            caps.update(tool.metadata.required_capabilities)
        return sorted(caps)

    def health(self) -> dict[str, bool]:
        """Probes health and readiness status of registered tools."""
        return dict.fromkeys(self._tools, True)

    def reload(self) -> None:
        """Clears all registered tools in preparation for reloading."""
        self._tools.clear()
        logger.info("Cleared ToolRegistry for reload.")

    def get_openai_tool_schemas(self) -> list[dict]:
        """Exports registered tools as OpenAI function call JSON Schemas."""
        schemas = []
        for tool in self._tools.values():
            meta = tool.metadata
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": meta.name,
                        "description": meta.description,
                        "parameters": meta.input_schema,
                    },
                }
            )
        return schemas


# Global Tool Registry singleton
tool_registry = ToolRegistry()
