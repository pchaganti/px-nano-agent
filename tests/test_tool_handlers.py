"""Tests for TodoWriteTool state management (merged from TodoManager)."""

import asyncio
from io import StringIO
from unittest.mock import patch

from nano_agent.tools import Todo, TodoStatus, TodoWriteTool


class TestTodoStatus:
    def test_status_values(self) -> None:
        assert TodoStatus.PENDING.value == "pending"
        assert TodoStatus.IN_PROGRESS.value == "in_progress"
        assert TodoStatus.COMPLETED.value == "completed"


class TestTodo:
    def test_from_dict(self) -> None:
        data = {
            "content": "Test task",
            "status": "in_progress",
            "activeForm": "Testing task",
        }
        todo = Todo.from_dict(data)
        assert todo.content == "Test task"
        assert todo.status == TodoStatus.IN_PROGRESS
        assert todo.active_form == "Testing task"

    def test_from_dict_defaults(self) -> None:
        todo = Todo.from_dict({})
        assert todo.content == ""
        assert todo.status == TodoStatus.PENDING
        assert todo.active_form == ""

    def test_to_dict(self) -> None:
        todo = Todo(
            content="Test task",
            status=TodoStatus.COMPLETED,
            active_form="Testing task",
        )
        result = todo.to_dict()
        assert result == {
            "content": "Test task",
            "status": "completed",
            "activeForm": "Testing task",
        }


class TestTodoWriteTool:
    """Tests for TodoWriteTool with integrated state management."""

    def test_init_empty(self) -> None:
        tool = TodoWriteTool()
        assert tool.todos == []

    def test_call_updates_state(self) -> None:
        tool = TodoWriteTool()
        todos_data = {
            "todos": [
                {
                    "content": "Task 1",
                    "status": "pending",
                    "activeForm": "Doing task 1",
                },
                {
                    "content": "Task 2",
                    "status": "completed",
                    "activeForm": "Doing task 2",
                },
            ]
        }
        with patch("sys.stdout", new_callable=StringIO):
            asyncio.run(tool.execute(todos_data))

        assert len(tool.todos) == 2
        assert tool.todos[0].content == "Task 1"
        assert tool.todos[0].status == TodoStatus.PENDING
        assert tool.todos[1].content == "Task 2"
        assert tool.todos[1].status == TodoStatus.COMPLETED

    def test_get_current_task(self) -> None:
        tool = TodoWriteTool()
        todos_data = {
            "todos": [
                {
                    "content": "Task 1",
                    "status": "completed",
                    "activeForm": "Doing task 1",
                },
                {
                    "content": "Task 2",
                    "status": "in_progress",
                    "activeForm": "Doing task 2",
                },
                {
                    "content": "Task 3",
                    "status": "pending",
                    "activeForm": "Doing task 3",
                },
            ]
        }
        with patch("sys.stdout", new_callable=StringIO):
            asyncio.run(tool.execute(todos_data))

        current = tool.get_current_task()
        assert current is not None
        assert current.content == "Task 2"
        assert current.status == TodoStatus.IN_PROGRESS

    def test_get_current_task_none(self) -> None:
        tool = TodoWriteTool()
        todos_data = {
            "todos": [
                {
                    "content": "Task 1",
                    "status": "completed",
                    "activeForm": "Doing task 1",
                },
                {
                    "content": "Task 2",
                    "status": "pending",
                    "activeForm": "Doing task 2",
                },
            ]
        }
        with patch("sys.stdout", new_callable=StringIO):
            asyncio.run(tool.execute(todos_data))

        current = tool.get_current_task()
        assert current is None

    def test_todos_returns_copy(self) -> None:
        tool = TodoWriteTool()
        todos_data = {
            "todos": [
                {
                    "content": "Task 1",
                    "status": "pending",
                    "activeForm": "Doing task 1",
                },
            ]
        }
        with patch("sys.stdout", new_callable=StringIO):
            asyncio.run(tool.execute(todos_data))

        todos_copy = tool.todos
        # Verify it's a copy, not the internal list
        assert todos_copy == tool.todos
        assert todos_copy is not tool._todos

    def test_clear(self) -> None:
        tool = TodoWriteTool()
        todos_data = {
            "todos": [
                {
                    "content": "Task 1",
                    "status": "pending",
                    "activeForm": "Doing task 1",
                },
            ]
        }
        with patch("sys.stdout", new_callable=StringIO):
            asyncio.run(tool.execute(todos_data))

        assert len(tool.todos) == 1
        tool.clear()
        assert len(tool.todos) == 0

    def test_call_returns_tool_result_message(self) -> None:
        tool = TodoWriteTool()
        tool_input = {
            "todos": [
                {"content": "Task 1", "status": "in_progress", "activeForm": "Doing 1"},
                {"content": "Task 2", "status": "pending", "activeForm": "Doing 2"},
            ]
        }
        with patch("sys.stdout", new_callable=StringIO):
            result = asyncio.run(tool.execute(tool_input))

        assert not isinstance(result, list)
        assert result.text == TodoWriteTool.TOOL_RESULT_MESSAGE
        assert len(tool.todos) == 2
        assert tool.todos[0].content == "Task 1"

    @patch("sys.stdout", new_callable=StringIO)
    def test_display_empty(self, mock_stdout: StringIO) -> None:
        tool = TodoWriteTool()
        tool.display()
        output = mock_stdout.getvalue()
        assert "No todos." in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_display_with_todos(self, mock_stdout: StringIO) -> None:
        tool = TodoWriteTool()
        todos_data = {
            "todos": [
                {
                    "content": "Task 1",
                    "status": "completed",
                    "activeForm": "Doing task 1",
                },
                {
                    "content": "Task 2",
                    "status": "in_progress",
                    "activeForm": "Doing task 2",
                },
                {
                    "content": "Task 3",
                    "status": "pending",
                    "activeForm": "Doing task 3",
                },
            ]
        }
        asyncio.run(tool.execute(todos_data))
        output = mock_stdout.getvalue()

        assert "--- Todo List ---" in output
        assert "[x] Task 1" in output
        assert "[~] Task 2" in output
        assert "[ ] Task 3" in output
        assert "Doing task 2..." in output
        assert "1/3 completed" in output
        assert "1 in progress" in output
        assert "1 pending" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_display_all_completed(self, mock_stdout: StringIO) -> None:
        tool = TodoWriteTool()
        todos_data = {
            "todos": [
                {
                    "content": "Task 1",
                    "status": "completed",
                    "activeForm": "Doing task 1",
                },
                {
                    "content": "Task 2",
                    "status": "completed",
                    "activeForm": "Doing task 2",
                },
            ]
        }
        asyncio.run(tool.execute(todos_data))
        output = mock_stdout.getvalue()

        assert "2/2 completed" in output
        assert "in progress" not in output
        assert "pending" not in output
