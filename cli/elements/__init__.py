"""Interactive UI elements for the CLI.

This module provides an abstraction for interactive terminal elements
that can control output and capture input.

Usage:
    from cli.elements import TerminalFooter, FooterElementManager, FooterInput

    footer = TerminalFooter()
    footer.activate()
    manager = FooterElementManager(footer)
    text = await manager.run(FooterInput(prompt="> "))
"""

from .base import ActiveElement, InputEvent
from .cancellation_menu import CancellationMenu
from .confirm_prompt import ConfirmPrompt
from .footer import StatusBarState, TerminalFooter
from .footer_input import FooterInput
from .footer_manager import FooterElementManager
from .menu_select import MenuSelect
from .terminal import RawInputReader, TerminalRegion

__all__ = [
    # Base
    "ActiveElement",
    "InputEvent",
    # Managers
    "FooterElementManager",
    # Footer
    "TerminalFooter",
    "StatusBarState",
    # Terminal
    "TerminalRegion",
    "RawInputReader",
    # Elements
    "FooterInput",
    "ConfirmPrompt",
    "MenuSelect",
    "CancellationMenu",
]
