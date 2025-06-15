#!/usr/bin/env python3
"""
Context Utilities

Utility functions for handling smart home context data:
- Loading and saving context files
- Deep merging context updates
- Context validation and normalization
"""

import json
from typing import Dict, Any, Union
from copy import deepcopy
from pathlib import Path

def load_context(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load smart home context from a JSON file.
    
    Args:
        file_path: Path to the context JSON file
        
    Returns:
        Context dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Context file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        context = json.load(f)
    
    return context

def save_context(context: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save smart home context to a JSON file.
    
    Args:
        context: Context dictionary to save
        file_path: Path where to save the context
    """
    file_path = Path(file_path)
    
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(context, f, indent=2)

def deep_merge_context(base_context: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge context updates into base context.
    
    Args:
        base_context: Original context dictionary
        updates: Updates to apply
        
    Returns:
        Merged context dictionary
    """
    # Create a deep copy to avoid modifying the original
    result = deepcopy(base_context)
    
    def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge source into target.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                _deep_merge(target[key], value)
            else:
                # Overwrite or add new key
                target[key] = deepcopy(value)
    
    _deep_merge(result, updates)
    return result

def validate_context(context: Dict[str, Any]) -> bool:
    """
    Validate that a context dictionary has the expected structure.
    
    Args:
        context: Context dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['time', 'sensors', 'devices']
    
    # Check if all required sections are present
    for section in required_sections:
        if section not in context:
            return False
    
    # Check if sensors and devices are dictionaries
    if not isinstance(context['sensors'], dict) or not isinstance(context['devices'], dict):
        return False
    
    return True

def normalize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize context dictionary to ensure consistent format.
    
    Args:
        context: Context dictionary to normalize
        
    Returns:
        Normalized context dictionary
    """
    normalized = deepcopy(context)
    
    # Ensure required sections exist
    if 'sensors' not in normalized:
        normalized['sensors'] = {}
    if 'devices' not in normalized:
        normalized['devices'] = {}
    if 'weather' not in normalized:
        normalized['weather'] = {}
    
    # Normalize boolean values
    bool_fields = ['user_home', 'user_asleep']
    for field in bool_fields:
        if field in normalized:
            if isinstance(normalized[field], str):
                normalized[field] = normalized[field].lower() in ['true', '1', 'yes', 'on']
    
    # Normalize device and sensor states
    for section in ['sensors', 'devices']:
        if section in normalized and isinstance(normalized[section], dict):
            for key, value in normalized[section].items():
                if isinstance(value, str):
                    # Normalize to uppercase for consistency
                    normalized[section][key] = value.upper()
    
    return normalized

def create_sample_context() -> Dict[str, Any]:
    """
    Create a sample context for testing purposes.
    
    Returns:
        Sample context dictionary
    """
    return {
        "time": "14:30",
        "user_home": True,
        "user_asleep": False,
        "aqi_outdoor": 85,
        "weather": {
            "temperature_outdoor": 22,
            "humidity_outdoor": 65,
            "condition": "sunny"
        },
        "sensors": {
            "gas_sensor": "OFF",
            "motion_bathroom": "OFF",
            "motion_living_room": "ON",
            "window_bedroom": "CLOSED",
            "window_kitchen": "CLOSED",
            "door_main": "LOCKED",
            "door_balcony": "CLOSED",
            "presence_sensor": "PRESENT"
        },
        "devices": {
            "smoke_detector": "ON",
            "ac_bedroom": "OFF",
            "heater_living_room": "OFF",
            "washing_machine": "OFF",
            "fan_kitchen": "OFF",
            "light_living_room": "ON",
            "tv_living_room": "OFF",
            "camera": "ON"
        }
    }

def extract_context_summary(context: Dict[str, Any]) -> str:
    """
    Extract a human-readable summary of the context.
    
    Args:
        context: Context dictionary
        
    Returns:
        Summary string
    """
    summary_parts = []
    
    # Time and user status
    time = context.get('time', 'unknown')
    user_home = context.get('user_home', False)
    user_asleep = context.get('user_asleep', False)
    
    summary_parts.append(f"Time: {time}")
    summary_parts.append(f"User: {'home' if user_home else 'away'}{', asleep' if user_asleep else ''}")
    
    # Weather
    weather = context.get('weather', {})
    if weather:
        temp = weather.get('temperature_outdoor', 'unknown')
        condition = weather.get('condition', 'unknown')
        summary_parts.append(f"Weather: {temp}°C, {condition}")
    
    # Air quality
    aqi = context.get('aqi_outdoor')
    if aqi is not None:
        aqi_level = "good" if aqi < 50 else "moderate" if aqi < 100 else "unhealthy" if aqi < 200 else "very unhealthy"
        summary_parts.append(f"AQI: {aqi} ({aqi_level})")
    
    # Active devices
    devices = context.get('devices', {})
    active_devices = [name for name, state in devices.items() if state == 'ON']
    if active_devices:
        summary_parts.append(f"Active devices: {', '.join(active_devices)}")
    
    # Open windows/doors
    sensors = context.get('sensors', {})
    open_items = [name for name, state in sensors.items() if state in ['OPEN', 'UNLOCKED']]
    if open_items:
        summary_parts.append(f"Open: {', '.join(open_items)}")
    
    return " | ".join(summary_parts)

def compare_contexts(context1: Dict[str, Any], context2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two contexts and return the differences.
    
    Args:
        context1: First context (original)
        context2: Second context (updated)
        
    Returns:
        Dictionary containing the differences
    """
    differences = {}
    
    def _compare_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = "") -> None:
        """
        Recursively compare dictionaries and track differences.
        """
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                differences[current_path] = {"added": dict2[key]}
            elif key not in dict2:
                differences[current_path] = {"removed": dict1[key]}
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    _compare_dicts(dict1[key], dict2[key], current_path)
                else:
                    differences[current_path] = {
                        "from": dict1[key],
                        "to": dict2[key]
                    }
    
    _compare_dicts(context1, context2)
    return differences