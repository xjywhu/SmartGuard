# Smart Home Risk Assessment System

A modular LLM-based system for assessing risks in smart home command execution. This system processes natural language commands through a pipeline of specialized modules to evaluate safety risks and make execution decisions.

## Architecture

The system consists of four main components:

1. **Command Parser** (LLM-based): Extracts device and action from natural language commands
2. **State Forecaster** (LLM-based): Predicts how the smart home context will change after command execution
3. **Risk Evaluator** (LLM-based): Assesses safety risks based on command and context
4. **Decision Module**: Makes execution decisions based on risk levels

## Features

- 🤖 **LLM-Powered**: Uses OpenAI GPT models for intelligent parsing and risk assessment
- 🔄 **Fallback Support**: Rule-based fallbacks when LLM is unavailable
- 📊 **Dataset Testing**: Batch testing against risk assessment datasets
- 🛡️ **Safety-First**: Prioritizes safety with comprehensive risk evaluation
- 🔧 **Modular Design**: Easy to extend and customize individual components

## Installation

1. **Clone or copy the system files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key** (optional but recommended):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Command Line Interface

#### Test Against Dataset
```bash
python main.py --test-dataset ../risky_smart_home_commands_dataset.json
```

#### Process Single Command
```bash
python main.py --command "Turn off the smoke detector" --context-file sample_context.json
```

#### Save Results
```bash
python main.py --test-dataset ../risky_smart_home_commands_dataset.json --output results.json
```

### Python API

```python
from smart_home_risk_system import SmartHomeRiskSystem
from smart_home_risk_system.utils import create_sample_context

# Initialize the system
system = SmartHomeRiskSystem()

# Create or load context
context = create_sample_context()

# Process a command
result = system.process_command("Turn off the smoke detector", context)

print(f"Risk Level: {result['risk_assessment']['risk_level']}")
print(f"Decision: {result['decision']['action']}")
print(f"Rationale: {result['risk_assessment']['rationale']}")
```

## Context Format

The system expects context in the following JSON format:

```json
{
  "time": "14:30",
  "user_home": true,
  "user_asleep": false,
  "aqi_outdoor": 85,
  "weather": {
    "temperature_outdoor": 22,
    "humidity_outdoor": 65,
    "condition": "sunny"
  },
  "sensors": {
    "gas_sensor": "OFF",
    "motion_bathroom": "OFF",
    "window_bedroom": "CLOSED",
    "door_main": "LOCKED"
  },
  "devices": {
    "smoke_detector": "ON",
    "ac_bedroom": "OFF",
    "heater_living_room": "OFF",
    "light_living_room": "ON"
  }
}
```

## Risk Levels

- **HIGH**: Immediate safety hazards, command will be blocked
- **MEDIUM**: Potential issues or inefficiencies, requires confirmation
- **LOW**: Safe operation, command allowed

## Example Output

```json
{
  "command": "Turn off the smoke detector",
  "parsed_command": {
    "device": "smoke_detector",
    "action": "OFF"
  },
  "risk_assessment": {
    "risk_level": "HIGH",
    "rationale": "Turning off smoke detector while gas sensor is active poses severe safety risks."
  },
  "decision": {
    "action": "BLOCKED",
    "rationale": "Command blocked due to high risk level"
  },
  "status": "completed"
}
```

## Configuration

### LLM Models

By default, the system uses `gpt-3.5-turbo`. You can configure different models:

```python
from smart_home_risk_system.modules import CommandParser, StateForecaster, RiskEvaluator

# Use GPT-4 for better accuracy
parser = CommandParser(model="gpt-4")
forecaster = StateForecaster(model="gpt-4")
evaluator = RiskEvaluator(model="gpt-4")
```

### Fallback Mode

If OpenAI API is not available, the system automatically falls back to rule-based processing:

- Command parsing uses keyword matching
- State forecasting uses simple device state updates
- Risk evaluation uses predefined safety rules

## Testing

Run the system against the provided dataset:

```bash
python main.py --test-dataset ../risky_smart_home_commands_dataset.json
```

This will:
- Process each command in the dataset
- Compare predicted vs expected risk levels
- Calculate accuracy metrics
- Display detailed results

## Extending the System

### Adding New Device Types

1. Update device mappings in `modules/command_parser.py`
2. Add state update rules in `modules/state_forecaster.py`
3. Include risk assessment rules in `modules/risk_evaluator.py`

### Custom Risk Rules

Add custom risk evaluation logic in `RiskEvaluator._fallback_evaluate()`:

```python
# Custom high-risk condition
if (device == "custom_device" and action == "dangerous_action"):
    return {
        "risk_level": "HIGH",
        "rationale": "Custom safety rule triggered"
    }
```

### Alternative LLM Providers

The system can be extended to use other LLM providers by modifying the API calls in each module.

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**
   - Set the `OPENAI_API_KEY` environment variable
   - System will fall back to rule-based processing

2. **Invalid Context Format**
   - Ensure context follows the expected JSON structure
   - Use `validate_context()` to check format

3. **LLM Response Parsing Errors**
   - System automatically falls back to rule-based processing
   - Check API response format if issues persist

### Debug Mode

Enable verbose output by modifying the print statements in `main.py` or add logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance

- **LLM Mode**: ~2-3 seconds per command (depends on API latency)
- **Fallback Mode**: ~0.1 seconds per command
- **Batch Processing**: Processes datasets efficiently with progress tracking

## Security

- API keys are loaded from environment variables
- No sensitive data is logged or stored
- Context data remains local unless explicitly saved

## Contributing

1. Follow the modular architecture
2. Add comprehensive error handling
3. Include fallback mechanisms for reliability
4. Test with various command types and contexts
5. Update documentation for new features

## License

This project is part of the SmartGuard smart home security research system.