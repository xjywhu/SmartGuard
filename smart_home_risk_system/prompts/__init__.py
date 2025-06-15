#!/usr/bin/env python3
"""
Prompts Package

LLM prompt templates and examples for the smart home risk assessment system.
"""

from .llm_templates import (
    COMMAND_PARSER_PROMPT,
    COMMAND_PARSER_EXAMPLES,
    STATE_FORECASTER_PROMPT,
    STATE_FORECASTER_EXAMPLES,
    RISK_EVALUATOR_PROMPT,
    RISK_EVALUATOR_EXAMPLES
)

__all__ = [
    "COMMAND_PARSER_PROMPT",
    "COMMAND_PARSER_EXAMPLES",
    "STATE_FORECASTER_PROMPT",
    "STATE_FORECASTER_EXAMPLES",
    "RISK_EVALUATOR_PROMPT",
    "RISK_EVALUATOR_EXAMPLES"
]