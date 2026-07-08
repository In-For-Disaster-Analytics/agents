"""LangSmith evaluation harness for the CKAN metadata LangGraph agent.

This package compares prompt/model configurations of the agent on a regression
suite, scoring flexible metadata fields (title, description) for reasonableness,
faithfulness, specificity, search usefulness, and catalog usability, and scoring
the agent's tool trajectory for safety and correctness.

See README.md in this directory for the full workflow.
"""

from __future__ import annotations

__all__ = ["__doc__"]
