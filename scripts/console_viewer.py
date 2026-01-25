"""Console-based visualization of conversation graphs."""

import json
import sys
from typing import Any


def truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def format_node(node: dict[str, Any]) -> tuple[str, str]:
    """Format a node for display. Returns (type_label, content)."""
    data = node["data"]
    node_id = node["id"][:6]

    if data.get("type") == "system_prompt":
        return "SYSTEM", truncate(data["content"])

    elif data.get("type") == "tool_definitions":
        tool_names = [t["name"] for t in data["tools"]]
        return "TOOLS", ", ".join(tool_names)

    elif data.get("type") == "tool_execution":
        tool_name = data["tool_name"]
        result = truncate(data["result"], 30)
        return "EXEC", f"{tool_name} -> {result}"

    elif data.get("type") == "stop_reason":
        return "STOP", f"{data['reason']}"

    elif data.get("role") == "user":
        content = data["content"]
        if isinstance(content, str):
            return "USER", truncate(content)
        elif isinstance(content, list):
            # Tool results
            tool_results = [c for c in content if c.get("type") == "tool_result"]
            if tool_results:
                return "RESULTS", f"({len(tool_results)} tool results)"
            return "USER", "(complex content)"

    elif data.get("role") == "assistant":
        content = data["content"]
        if isinstance(content, str):
            return "ASSISTANT", truncate(content)
        elif isinstance(content, list):
            # Check for tool_use blocks
            tool_uses = [c for c in content if c.get("type") == "tool_use"]
            text_blocks = [c for c in content if c.get("type") == "text"]

            if tool_uses and not text_blocks:
                tools = [f"{t['name']}(...)" for t in tool_uses]
                if len(tools) > 1:
                    return "TOOL_USE", " + ".join(tools)
                return "TOOL_USE", tools[0]
            elif text_blocks:
                text = text_blocks[0].get("text", "")
                return "ASSISTANT", truncate(text)
            return "ASSISTANT", "(thinking)"

    return "UNKNOWN", str(data)[:40]


def render_graph(graph_path: str) -> None:
    """Render graph to console."""
    with open(graph_path) as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    head_ids = graph["head_ids"]

    # Build parent->children map
    children_map: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for pid in node["parent_ids"]:
            if pid in children_map:
                children_map[pid].append(nid)

    # Find root nodes (no parents)
    roots = [nid for nid, node in nodes.items() if not node["parent_ids"]]

    # Build display order via BFS
    visited: set[str] = set()
    display_order: list[str] = []

    def visit(nid: str) -> None:
        if nid in visited:
            return
        # Check all parents visited
        node = nodes[nid]
        for pid in node["parent_ids"]:
            if pid not in visited:
                return
        visited.add(nid)
        display_order.append(nid)
        for child_id in children_map[nid]:
            visit(child_id)

    # Start from roots
    queue = list(roots)
    while queue:
        nid = queue.pop(0)
        visit(nid)
        for child_id in children_map.get(nid, []):
            if child_id not in visited:
                queue.append(child_id)

    # Print header
    print()
    print("=" * 75)
    print(f"  CONVERSATION GRAPH ({len(nodes)} nodes)".center(75))
    print("=" * 75)
    print()

    # Track parallel executions for visualization
    i = 0
    while i < len(display_order):
        nid = display_order[i]
        node = nodes[nid]
        type_label, content = format_node(node)
        short_id = nid[:6]

        # Check if this node has multiple children (fan-out)
        children = children_map[nid]
        is_fanout = len(children) > 1

        # Check if this node has multiple parents (fan-in / merge)
        is_fanin = len(node["parent_ids"]) > 1

        # Color codes
        colors = {
            "SYSTEM": "\033[36m",  # Cyan
            "TOOLS": "\033[35m",  # Magenta
            "USER": "\033[32m",  # Green
            "ASSISTANT": "\033[33m",  # Yellow
            "TOOL_USE": "\033[34m",  # Blue
            "EXEC": "\033[90m",  # Gray
            "RESULTS": "\033[32m",  # Green
            "STOP": "\033[31m",  # Red
        }
        reset = "\033[0m"
        color = colors.get(type_label, "")

        # Print the node
        if is_fanin:
            print("    └────────┬─────────┘")
            print("             │")

        print(f"  {color}[{short_id}] {type_label}: {content}{reset}")

        # Handle fan-out (parallel execution)
        if is_fanout:
            print("    │")
            print("    ├" + "─" * 18 + "┐")

            # Print parallel children side by side
            parallel_nodes = []
            for child_id in children:
                child_node = nodes[child_id]
                child_type, child_content = format_node(child_node)
                parallel_nodes.append((child_id[:6], child_type, child_content))

            # Print each parallel branch
            for j, (cid, ctype, ccontent) in enumerate(parallel_nodes):
                prefix = "    │" if j == 0 else "                   │"
                color = colors.get(ctype, "")
                if j == 0:
                    print(f"    ▼                  ▼")
                    left = f"[{parallel_nodes[0][0]}] {parallel_nodes[0][1]}"
                    right = f"[{parallel_nodes[1][0]}] {parallel_nodes[1][1]}"
                    print(
                        f"  {colors.get(parallel_nodes[0][1], '')}{left:18}{reset} {colors.get(parallel_nodes[1][1], '')}{right}{reset}"
                    )
                    left_c = truncate(parallel_nodes[0][2], 16)
                    right_c = truncate(parallel_nodes[1][2], 20)
                    print(f"   {left_c:18}  {right_c}")
                    break

            print("    │                  │")

            # Skip the children we just printed
            for child_id in children:
                if child_id in display_order:
                    idx = display_order.index(child_id)
                    if idx > i:
                        display_order.remove(child_id)

        elif nid not in head_ids:
            print("    │")
            print("    ▼")

        i += 1

    # Print end marker
    print()
    print("=" * 75)
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python console_viewer.py <graph.json>")
        sys.exit(1)

    render_graph(sys.argv[1])


if __name__ == "__main__":
    main()
