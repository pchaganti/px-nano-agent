"""Edit tool for file modifications with preview confirmation."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from ..data_structures import TextContent
from .base import Desc, Tool

# =============================================================================
# Module-level state for Edit tool confirmation workflow
# =============================================================================


@dataclass
class PendingEdit:
    """Stores a pending edit awaiting confirmation."""

    file_path: str
    old_string: str
    new_string: str
    match_line: int
    context_before: list[str]  # Lines before match
    context_after: list[str]  # Lines after match
    created_at: float
    replace_all: bool = False
    match_count: int = 1


# Global dict to store pending edits by edit_id
_pending_edits: dict[str, PendingEdit] = {}

# Expiry time for pending edits (5 minutes)
_EDIT_EXPIRY_SECONDS = 300


def get_pending_edit(edit_id: str) -> PendingEdit | None:
    """Get a pending edit by ID for display purposes.

    Cleans up expired edits before returning.

    Args:
        edit_id: The edit ID to look up.

    Returns:
        The PendingEdit if found and not expired, None otherwise.
    """
    # Cleanup expired edits first
    current_time = time.time()
    expired = [
        k
        for k, v in _pending_edits.items()
        if current_time - v.created_at > _EDIT_EXPIRY_SECONDS
    ]
    for k in expired:
        del _pending_edits[k]
    return _pending_edits.get(edit_id)


def _cleanup_expired_edits() -> None:
    """Remove expired pending edits."""
    current_time = time.time()
    expired = [
        edit_id
        for edit_id, edit in _pending_edits.items()
        if current_time - edit.created_at > _EDIT_EXPIRY_SECONDS
    ]
    for edit_id in expired:
        del _pending_edits[edit_id]


def _find_match_line(content: str, old_string: str) -> tuple[int, int]:
    """Find the line number where old_string starts.

    Returns:
        Tuple of (line_number, char_position) where line_number is 0-indexed.
    """
    pos = content.find(old_string)
    if pos == -1:
        return (-1, -1)
    # Count newlines before the match position
    line_num = content[:pos].count("\n")
    return (line_num, pos)


def _generate_preview(
    lines: list[str],
    match_line: int,
    old_string: str,
    new_string: str,
    context_lines: int = 3,
) -> tuple[str, list[str], list[str]]:
    """Generate a diff-style preview of the edit.

    Returns:
        Tuple of (preview_text, context_before, context_after)
    """
    total_lines = len(lines)

    # Get context lines
    start = max(0, match_line - context_lines)
    end = min(total_lines, match_line + context_lines + old_string.count("\n") + 1)

    context_before = lines[start:match_line]
    context_after = lines[match_line + old_string.count("\n") + 1 : end]

    # Build preview
    preview_parts = []

    # Show lines before
    for i, line in enumerate(lines[start:match_line], start=start + 1):
        preview_parts.append(f"     {i:6}\t{line}")

    # Show old lines (to be removed)
    old_lines = old_string.splitlines()
    for i, line in enumerate(old_lines):
        line_num = match_line + i + 1
        preview_parts.append(f"  -  {line_num:6}\t{line}")

    # Show new lines (to be added)
    new_lines = new_string.splitlines()
    for i, line in enumerate(new_lines):
        line_num = match_line + i + 1
        preview_parts.append(f"  +  {line_num:6}\t{line}")

    # Show lines after
    after_start = match_line + len(old_lines)
    for i, line in enumerate(lines[after_start:end], start=after_start + 1):
        preview_parts.append(f"     {i:6}\t{line}")

    return "\n".join(preview_parts), context_before, context_after


# =============================================================================
# Input Dataclasses
# =============================================================================


@dataclass
class EditInput:
    """Input for EditTool."""

    file_path: Annotated[str, Desc("The absolute path to the file to modify")]
    old_string: Annotated[str, Desc("The text to replace")]
    new_string: Annotated[
        str, Desc("The text to replace it with (must be different from old_string)")
    ]
    replace_all: Annotated[
        bool, Desc("Replace all occurences of old_string (default false)")
    ] = False


@dataclass
class EditConfirmInput:
    """Input for EditConfirmTool."""

    edit_id: Annotated[str, Desc("The edit ID from preview to confirm or reject")]


# =============================================================================
# Tool Classes
# =============================================================================


@dataclass
class EditTool(Tool):
    """Performs exact string replacements in files with two-step confirmation.

    This tool uses a preview → confirm workflow to prevent hallucination:
    1. Step 1 (Preview): Returns a diff preview, stores pending edit
    2. Step 2 (Confirm): Use EditConfirmTool to apply or reject

    This ensures the agent sees actual file content before changes are made.
    """

    name: str = "Edit"
    description: str = """Performs exact string replacements in files with preview confirmation.

Usage:
- You must use your Read tool at least once in the conversation before editing.
- Returns a PREVIEW of changes - edit is NOT applied immediately.
- After reviewing preview, call EditConfirm(edit_id="...") to apply.
- The edit will FAIL if old_string is not unique in the file (unless replace_all=True).
- Use replace_all=True for replacing all occurrences across the file.

Workflow:
1. Call Edit(file_path, old_string, new_string) → Returns preview with edit_id
2. Review the preview to verify it matches expectations
3. Call EditConfirm(edit_id="...") to apply, or let it expire to reject"""

    async def __call__(self, input: EditInput) -> TextContent:
        """Generate edit preview and store pending edit for confirmation."""
        # Cleanup expired edits first
        _cleanup_expired_edits()

        path = Path(input.file_path)

        # Validate file exists
        if not path.exists():
            return TextContent(text=f"Error: File not found: {input.file_path}")
        if not path.is_file():
            return TextContent(text=f"Error: Not a file: {input.file_path}")

        try:
            content = path.read_text()
        except PermissionError:
            return TextContent(text=f"Error: Permission denied: {input.file_path}")
        except UnicodeDecodeError:
            return TextContent(
                text=f"Error: Cannot edit binary file: {input.file_path}"
            )

        # Validate old_string exists
        if input.old_string not in content:
            return TextContent(
                text=f"Error: old_string not found in {input.file_path}\n\n"
                "The text you're trying to replace does not exist in the file. "
                "Use the Read tool to verify the exact content."
            )

        # Check uniqueness (unless replace_all is True)
        match_count = content.count(input.old_string)
        if match_count > 1 and not input.replace_all:
            return TextContent(
                text=f"Error: old_string is not unique in {input.file_path}\n\n"
                f"Found {match_count} occurrences. Either:\n"
                "1. Provide more context to make old_string unique, or\n"
                "2. Set replace_all=True to replace all occurrences."
            )

        # Validate old_string != new_string
        if input.old_string == input.new_string:
            return TextContent(
                text="Error: old_string and new_string are identical. No changes to make."
            )

        # Find match location and generate preview
        lines = content.splitlines()
        match_line, _ = _find_match_line(content, input.old_string)

        preview, context_before, context_after = _generate_preview(
            lines, match_line, input.old_string, input.new_string
        )

        # Generate unique edit_id and store pending edit
        edit_id = str(uuid.uuid4())[:8]
        _pending_edits[edit_id] = PendingEdit(
            file_path=input.file_path,
            old_string=input.old_string,
            new_string=input.new_string,
            match_line=match_line,
            context_before=context_before,
            context_after=context_after,
            created_at=time.time(),
            replace_all=input.replace_all,
            match_count=match_count,
        )

        # Build output
        header = f"─── Edit Preview: {input.file_path} ───\n"
        if match_count > 1:
            header += f"Replacing ALL {match_count} occurrences\n"
        else:
            header += f"Match found at line {match_line + 1}\n"
        header += "\n"

        footer = (
            f'\n\n⚠️ Edit NOT applied. Call EditConfirm(edit_id="{edit_id}") to apply if the edit matches what you want. Otherwise, ignore it.\n'
            f"Edit expires in {_EDIT_EXPIRY_SECONDS // 60} minutes."
        )

        return TextContent(text=header + preview + footer)


@dataclass
class EditConfirmTool(Tool):
    """Confirms and applies a pending edit from EditTool.

    This is the second step in the two-step edit confirmation workflow.
    Use the edit_id from the EditTool preview to apply the edit.
    """

    name: str = "EditConfirm"
    description: str = """Confirms and applies a pending edit from EditTool.

Usage:
- Provide the edit_id from the Edit tool preview
- The edit will be applied to the file
- If the edit_id is invalid or expired, an error is returned

Example:
  EditConfirm(edit_id="abc12345")"""

    async def __call__(self, input: EditConfirmInput) -> TextContent:
        """Apply a pending edit by its ID."""
        _cleanup_expired_edits()

        edit_id = input.edit_id

        if edit_id not in _pending_edits:
            return TextContent(
                text=f"Error: Edit ID '{edit_id}' not found or expired.\n\n"
                "Pending edits expire after 5 minutes. "
                "Use the Edit tool again to create a new preview."
            )

        pending = _pending_edits[edit_id]
        path = Path(pending.file_path)

        # Re-validate file exists
        if not path.exists():
            del _pending_edits[edit_id]
            return TextContent(
                text=f"Error: File no longer exists: {pending.file_path}"
            )

        try:
            content = path.read_text()
        except (PermissionError, UnicodeDecodeError) as e:
            del _pending_edits[edit_id]
            return TextContent(text=f"Error reading file: {e}")

        # Validate old_string still exists
        if pending.old_string not in content:
            del _pending_edits[edit_id]
            return TextContent(
                text=f"Error: The file has been modified since preview.\n"
                "The old_string no longer exists. Use Edit tool again."
            )

        # Apply the edit
        if pending.replace_all:
            new_content = content.replace(pending.old_string, pending.new_string)
        else:
            new_content = content.replace(pending.old_string, pending.new_string, 1)

        try:
            path.write_text(new_content)
        except PermissionError:
            del _pending_edits[edit_id]
            return TextContent(text=f"Error: Permission denied: {pending.file_path}")
        except Exception as e:
            del _pending_edits[edit_id]
            return TextContent(text=f"Error writing file: {e}")

        # Cleanup and confirm
        del _pending_edits[edit_id]

        if pending.replace_all and pending.match_count > 1:
            return TextContent(
                text=f"✓ Edit applied: {pending.file_path}\n"
                f"  Replaced {pending.match_count} occurrences."
            )
        else:
            return TextContent(
                text=f"✓ Edit applied: {pending.file_path}\n"
                f"  Modified at line {pending.match_line + 1}."
            )
