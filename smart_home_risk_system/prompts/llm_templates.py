#!/usr/bin/env python3
"""
LLM Prompt Templates

Contains system prompts and few-shot examples for each LLM-based module:
1. Command Parser
2. State Forecaster
3. Risk Evaluator
"""

# Command Parser Prompts
COMMAND_PARSER_PROMPT = """
You are a smart home command parser. Your task is to extract the device and intended action from natural language commands.

Return a JSON object with exactly two fields:
- "device": the target device (use snake_case, e.g., "smoke_detector")
- "action": the intended action (use uppercase: ON, OFF, INCREASE, DECREASE, SET, OPEN, CLOSE, etc.)

Be precise and consistent with device names and actions. If the command is unclear, make your best interpretation.
"""

COMMAND_PARSER_EXAMPLES = [
    {
        "input": "Turn off the smoke detector",
        "output": {"device": "smoke_detector", "action": "OFF"}
    },
    {
        "input": "Switch on the living room light",
        "output": {"device": "light_living_room", "action": "ON"}
    },
    {
        "input": "Open the bedroom window",
        "output": {"device": "window_bedroom", "action": "OPEN"}
    },
    {
        "input": "Start the air conditioner",
        "output": {"device": "air_conditioner", "action": "ON"}
    },
    {
        "input": "Increase the heater temperature",
        "output": {"device": "heater", "action": "INCREASE"}
    },
    {
        "input": "Turn on the TV",
        "output": {"device": "tv", "action": "ON"}
    },
    {
        "input": "Close the main door",
        "output": {"device": "door_main", "action": "CLOSE"}
    },
    {
        "input": "Activate the security camera",
        "output": {"device": "camera", "action": "ON"}
    }
]

# State Forecaster Prompts
STATE_FORECASTER_PROMPT = """
You are a smart home state forecaster. Given a command and the current context, predict what changes will occur in the smart home state after the command is executed.

Return a JSON object containing only the fields that will change. Use the same structure as the input context.
For example, if a device state changes, return: {"devices": {"device_name": "new_state"}}
If a sensor reading changes, return: {"sensors": {"sensor_name": "new_value"}}

Only include fields that actually change. Do not repeat unchanged values.
"""

STATE_FORECASTER_EXAMPLES = [
    {
        "command": {"device": "smoke_detector", "action": "OFF"},
        "current_context": {
            "devices": {
                "smoke_detector": "ON",
                "ac_bedroom": "OFF"
            }
        },
        "changes": {
            "devices": {
                "smoke_detector": "OFF"
            }
        }
    },
    {
        "command": {"device": "window_bedroom", "action": "OPEN"},
        "current_context": {
            "sensors": {
                "window_bedroom": "CLOSED",
                "motion_living_room": "OFF"
            }
        },
        "changes": {
            "sensors": {
                "window_bedroom": "OPEN"
            }
        }
    },
    {
        "command": {"device": "air_conditioner", "action": "ON"},
        "current_context": {
            "devices": {
                "ac_bedroom": "OFF",
                "heater_living_room": "OFF"
            }
        },
        "changes": {
            "devices": {
                "ac_bedroom": "ON"
            }
        }
    }
]

# Risk Evaluator Prompts
RISK_EVALUATOR_PROMPT = """
You are a smart home safety risk evaluator. Analyze whether executing a command in the given context poses any safety risks.

Consider factors like:
- Safety device status (smoke detectors, gas sensors)
- Environmental conditions (weather, air quality, temperature)
- User presence and state (home, asleep)
- Device interactions and conflicts
- Time of day appropriateness

Return a JSON object with:
- "risk_level": "LOW", "MEDIUM", or "HIGH"
- "rationale": detailed explanation of the risk assessment

HIGH risk: Immediate safety hazards, potential for harm
MEDIUM risk: Inefficiency, minor issues, or potential problems
LOW risk: Safe operation with minimal concerns
"""

RISK_EVALUATOR_EXAMPLES = [
    {
        "command": {"device": "smoke_detector", "action": "OFF"},
        "context": {
            "time": "14:15",
            "user_home": False,
            "sensors": {
                "gas_sensor": "ON"
            },
            "devices": {
                "smoke_detector": "ON",
                "oven": "ON"
            }
        },
        "risk_assessment": {
            "risk_level": "HIGH",
            "rationale": "Turning off the smoke detector while gas is detected and oven is running poses severe safety risks, especially with user not home."
        }
    },
    {
        "command": {"device": "window_bedroom", "action": "OPEN"},
        "context": {
            "time": "03:56",
            "user_asleep": True,
            "aqi_outdoor": 238,
            "weather": {
                "temperature_outdoor": 13,
                "condition": "rainy"
            }
        },
        "risk_assessment": {
            "risk_level": "MEDIUM",
            "rationale": "Opening window during poor air quality (AQI 238) and rain while user is asleep may cause health issues and water damage."
        }
    },
    {
        "command": {"device": "light_living_room", "action": "ON"},
        "context": {
            "time": "19:30",
            "user_home": True,
            "devices": {
                "light_living_room": "OFF"
            }
        },
        "risk_assessment": {
            "risk_level": "LOW",
            "rationale": "Turning on living room light in the evening while user is home is a normal, safe operation."
        }
    },
    {
        "command": {"device": "air_conditioner", "action": "ON"},
        "context": {
            "time": "15:00",
            "weather": {
                "temperature_outdoor": 35
            },
            "sensors": {
                "window_bedroom": "OPEN",
                "window_kitchen": "OPEN"
            },
            "devices": {
                "ac_bedroom": "OFF"
            }
        },
        "risk_assessment": {
            "risk_level": "MEDIUM",
            "rationale": "Running AC while windows are open wastes energy and reduces efficiency, though not a safety hazard."
        }
    },
    {
        "command": {"device": "heater", "action": "ON"},
        "context": {
            "time": "14:00",
            "user_home": False,
            "weather": {
                "temperature_outdoor": 38
            },
            "devices": {
                "heater_living_room": "OFF"
            }
        },
        "risk_assessment": {
            "risk_level": "HIGH",
            "rationale": "Turning on heater when outdoor temperature is 38°C and user is not home poses fire risk and extreme energy waste."
        }
    }
]