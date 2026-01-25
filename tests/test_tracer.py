"""Tests for Node-based conversation graph."""

import tempfile
from pathlib import Path

from nano_agent import (
    Message,
    Node,
    Role,
    SystemPrompt,
    TextContent,
    ToolDefinitions,
    ToolResultContent,
    ToolUseContent,
)


class TestNodeCreation:
    def test_system_prompt_node(self) -> None:
        node = Node.system("You are helpful.")
        assert node.is_root
        assert isinstance(node.data, SystemPrompt)
        assert node.data.content == "You are helpful."

    def test_tools_node(self) -> None:
        from nano_agent.tools import TodoWriteTool

        root = Node.system("Test")
        node = root.tools(TodoWriteTool())
        assert isinstance(node.data, ToolDefinitions)
        assert len(node.data.tools) == 1
        assert node.parents == (root,)

    def test_child_node(self) -> None:
        root = Node.system("Test")
        child = root.child(Message(Role.USER, "Hello"))
        assert child.parents == (root,)

    def test_with_parents(self) -> None:
        n1 = Node.root(Message(Role.USER, "A"))
        n2 = Node.root(Message(Role.USER, "B"))
        n3 = Node.with_parents([n1, n2], Message(Role.USER, "Merged"))
        assert n3.parents == (n1, n2)


class TestGraphTraversal:
    def test_ancestors(self) -> None:
        n1 = Node.system("Test")
        n2 = n1.child(Message(Role.USER, "Hello"))
        n3 = n2.child(Message(Role.ASSISTANT, "Hi"))

        ancestors = n3.ancestors()
        assert len(ancestors) == 3
        assert ancestors[0] == n1
        assert ancestors[2] == n3

    def test_to_messages(self) -> None:
        root = Node.system("Test")
        n1 = root.child(Message(Role.USER, "Hello"))
        n2 = n1.child(Message(Role.ASSISTANT, "Hi"))

        messages = n2.to_messages()
        assert len(messages) == 2  # Only Message nodes
        assert messages[0].content == "Hello"

    def test_get_system_prompt(self) -> None:
        root = Node.system("You are helpful.")
        n1 = root.child(Message(Role.USER, "Hi"))

        assert n1.get_system_prompt() == "You are helpful."

    def test_get_tools(self) -> None:
        from nano_agent.tools import TodoWriteTool

        root = Node.system("Test")
        tools_node = root.tools(TodoWriteTool())
        n1 = tools_node.child(Message(Role.USER, "Hi"))

        tools = n1.get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "TodoWrite"


class TestSerialization:
    def test_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"

            root = Node.system("You are helpful.")
            n1 = root.child(Message(Role.USER, "Hello"))
            n2 = n1.child(Message(Role.ASSISTANT, "Hi"))

            Node.save_graph(n2, path)
            heads, meta = Node.load_graph(path)

            assert len(heads) == 1
            assert heads[0].get_system_prompt() == "You are helpful."
            assert len(heads[0].to_messages()) == 2

    def test_save_with_tools(self) -> None:
        from nano_agent.tools import TodoWriteTool

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"

            root = Node.system("Test")
            tools_node = root.tools(TodoWriteTool())
            n1 = tools_node.child(Message(Role.USER, "Hi"))

            Node.save_graph(n1, path)
            heads, _ = Node.load_graph(path)

            tools = heads[0].get_tools()
            assert len(tools) == 1


class TestToolUse:
    def test_tool_content(self) -> None:
        root = Node.system("Test")
        n1 = root.child(Message(Role.USER, "List files"))

        tool_use = ToolUseContent(id="t1", name="bash", input={"cmd": "ls"})
        n2 = n1.child(Message(Role.ASSISTANT, [tool_use]))

        tool_result = ToolResultContent(
            tool_use_id="t1", content=[TextContent(text="file.txt")]
        )
        n3 = n2.child(Message(Role.USER, [tool_result]))

        assert len(n3.to_messages()) == 3


class TestRepr:
    def test_repr_message(self) -> None:
        node = Node.root(Message(Role.USER, "Hi"))
        assert "Message" in repr(node)

    def test_repr_system(self) -> None:
        node = Node.system("Test")
        assert "SystemPrompt" in repr(node)
