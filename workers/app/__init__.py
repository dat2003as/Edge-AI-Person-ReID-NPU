"""
App module - refactored person tracking application.

This module provides the main entry point for the person tracking system.
All app-related components are organized in this package.
"""

from workers.app.main import run_application

__all__ = ['run_application']
