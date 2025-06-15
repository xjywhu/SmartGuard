#!/usr/bin/env python3
"""
Enhanced Context Builder for Smart Home Risk Assessment

This module builds comprehensive context using:
1. Knowledge Graph representation of home systems
2. Graph Neural Network-inspired features
3. Temporal and spatial relationships
4. Device state dependencies
5. User behavior patterns
"""

import json
import os
from datetime import datetime, time
from typing import Dict, List, Any, Tuple
import networkx as nx
from collections import defaultdict

class EnhancedContextBuilder:
    """Enhanced context builder using knowledge graphs and GNN-inspired features."""
    
    def __init__(self, home_system_path: str = None):
        self.home_system_path = home_system_path or "../commandData/homeSystem.json"
        self.knowledge_graph = None
        self.device_embeddings = {}
        self.spatial_relationships = {}
        self.temporal_patterns = {}
        self.load_knowledge_graph()
        self.build_spatial_relationships()
        self.initialize_device_embeddings()
    
    def load_knowledge_graph(self):
        """Load and build knowledge graph from home system data."""
        try:
            # Try multiple possible paths
            possible_paths = [
                self.home_system_path,
                "../commandData/homeSystem.json",
                "../../commandData/homeSystem.json",
                "d:/SmartGuard/commandData/homeSystem.json"
            ]
            
            home_data = None
            for path in possible_paths:
                try:
                    with open(path, 'r') as f:
                        home_data = json.load(f)
                    print(f"Loaded home system from: {path}")
                    break
                except FileNotFoundError:
                    continue
            
            if not home_data:
                print("Warning: Could not load home system data, using minimal structure")
                home_data = self._create_minimal_home_structure()
            
            self.knowledge_graph = nx.Graph()
            self._build_graph_from_data(home_data)
            
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            self.knowledge_graph = nx.Graph()
            self._create_minimal_graph()
    
    def _create_minimal_home_structure(self) -> Dict:
        """Create a minimal home structure for testing."""
        return {
            "Home_System": {
                "Living_Room": {
                    "TV": {"States": ["on", "off"], "Default_State": "off"},
                    "Speaker": {"States": ["playing", "stopped"], "Default_State": "stopped"}
                },
                "Kitchen": {
                    "Oven": {"States": ["on", "off"], "Default_State": "off"},
                    "Refrigerator": {"States": ["running", "off"], "Default_State": "running"}
                },
                "Bedroom": {
                    "Light": {"States": ["on", "off"], "Default_State": "off"},
                    "AC": {"States": ["cooling", "heating", "off"], "Default_State": "off"}
                }
            }
        }
    
    def _build_graph_from_data(self, home_data: Dict):
        """Build NetworkX graph from home system data."""
        if "Home_System" not in home_data:
            return
        
        home_system = home_data["Home_System"]
        
        # Add nodes for rooms and devices
        for room_name, room_data in home_system.items():
            if room_name == "Whole_House_Systems":
                continue
                
            # Add room node
            self.knowledge_graph.add_node(room_name, type="room")
            
            # Add device nodes and connect to room
            for device_name, device_data in room_data.items():
                device_id = f"{room_name}_{device_name}"
                self.knowledge_graph.add_node(
                    device_id, 
                    type="device",
                    room=room_name,
                    device_type=device_name,
                    states=device_data.get("States", []),
                    default_state=device_data.get("Default_State", "unknown")
                )
                
                # Connect device to room
                self.knowledge_graph.add_edge(room_name, device_id, relation="contains")
    
    def _create_minimal_graph(self):
        """Create a minimal graph for testing."""
        rooms = ["Living_Room", "Kitchen", "Bedroom"]
        devices = {
            "Living_Room": ["TV", "Speaker"],
            "Kitchen": ["Oven", "Refrigerator"],
            "Bedroom": ["Light", "AC"]
        }
        
        for room in rooms:
            self.knowledge_graph.add_node(room, type="room")
            for device in devices[room]:
                device_id = f"{room}_{device}"
                self.knowledge_graph.add_node(device_id, type="device", room=room, device_type=device)
                self.knowledge_graph.add_edge(room, device_id, relation="contains")
    
    def build_spatial_relationships(self):
        """Build spatial relationships between rooms and devices."""
        # Define typical room adjacencies
        adjacencies = {
            "Living_Room": ["Kitchen", "Bedroom", "Dining_Room"],
            "Kitchen": ["Living_Room", "Dining_Room"],
            "Bedroom": ["Living_Room", "Bathroom"],
            "Bathroom": ["Bedroom", "Master_Bedroom"],
            "Study_Office": ["Living_Room", "Bedroom"]
        }
        
        # Add spatial edges to knowledge graph
        for room, adjacent_rooms in adjacencies.items():
            if self.knowledge_graph.has_node(room):
                for adj_room in adjacent_rooms:
                    if self.knowledge_graph.has_node(adj_room):
                        self.knowledge_graph.add_edge(room, adj_room, relation="adjacent")
        
        self.spatial_relationships = adjacencies
    
    def initialize_device_embeddings(self):
        """Initialize GNN-inspired device embeddings based on graph structure."""
        for node in self.knowledge_graph.nodes():
            node_data = self.knowledge_graph.nodes[node]
            
            if node_data.get("type") == "device":
                # Create embedding based on device properties
                embedding = {
                    "centrality": nx.degree_centrality(self.knowledge_graph)[node],
                    "room_connectivity": len(list(self.knowledge_graph.neighbors(node))),
                    "device_type_encoding": self._encode_device_type(node_data.get("device_type", "")),
                    "state_complexity": len(node_data.get("states", [])),
                    "safety_criticality": self._assess_safety_criticality(node_data.get("device_type", ""))
                }
                self.device_embeddings[node] = embedding
    
    def _encode_device_type(self, device_type: str) -> float:
        """Encode device type as numerical feature."""
        device_categories = {
            "security": 0.9,  # High importance
            "safety": 0.95,   # Highest importance
            "entertainment": 0.3,
            "climate": 0.7,
            "lighting": 0.4,
            "appliance": 0.6
        }
        
        device_type_lower = device_type.lower()
        
        if any(keyword in device_type_lower for keyword in ["camera", "lock", "alarm", "sensor"]):
            return device_categories["security"]
        elif any(keyword in device_type_lower for keyword in ["detector", "fire", "smoke", "carbon"]):
            return device_categories["safety"]
        elif any(keyword in device_type_lower for keyword in ["tv", "speaker", "radio"]):
            return device_categories["entertainment"]
        elif any(keyword in device_type_lower for keyword in ["ac", "heater", "thermostat"]):
            return device_categories["climate"]
        elif any(keyword in device_type_lower for keyword in ["light", "lamp"]):
            return device_categories["lighting"]
        else:
            return device_categories["appliance"]
    
    def _assess_safety_criticality(self, device_type: str) -> float:
        """Assess safety criticality of device type."""
        high_risk_devices = ["oven", "stove", "heater", "fireplace", "smoke_detector"]
        medium_risk_devices = ["door_lock", "security_camera", "garage_door"]
        
        device_type_lower = device_type.lower()
        
        if any(device in device_type_lower for device in high_risk_devices):
            return 0.9
        elif any(device in device_type_lower for device in medium_risk_devices):
            return 0.6
        else:
            return 0.3
    
    def build_enhanced_context(self, command: str, base_context: Dict = None) -> Dict:
        """Build enhanced context with knowledge graph and GNN features."""
        current_time = datetime.now()
        
        # Start with base context or create minimal one
        if base_context:
            enhanced_context = base_context.copy()
        else:
            enhanced_context = {
                "time": current_time.strftime("%H:%M"),
                "date": current_time.strftime("%Y-%m-%d"),
                "user_home": True,
                "user_asleep": self._is_sleep_time(current_time.time())
            }
        
        # Add knowledge graph features
        enhanced_context["knowledge_graph"] = self._extract_graph_features(command)
        
        # Add device state predictions
        enhanced_context["device_states"] = self._predict_device_states(command, current_time)
        
        # Add spatial context
        enhanced_context["spatial_context"] = self._build_spatial_context(command)
        
        # Add temporal patterns
        enhanced_context["temporal_patterns"] = self._analyze_temporal_patterns(current_time)
        
        # Add risk indicators
        enhanced_context["risk_indicators"] = self._extract_risk_indicators(command, enhanced_context)
        
        # Add device dependencies
        enhanced_context["device_dependencies"] = self._analyze_device_dependencies(command)
        
        return enhanced_context
    
    def _extract_graph_features(self, command: str) -> Dict:
        """Extract graph-based features for the command."""
        # Parse command to identify target device/room
        target_devices = self._identify_target_devices(command)
        
        graph_features = {
            "total_devices": len([n for n in self.knowledge_graph.nodes() if self.knowledge_graph.nodes[n].get("type") == "device"]),
            "total_rooms": len([n for n in self.knowledge_graph.nodes() if self.knowledge_graph.nodes[n].get("type") == "room"]),
            "target_devices": target_devices,
            "affected_devices": [],
            "network_density": nx.density(self.knowledge_graph)
        }
        
        # Find devices that might be affected
        for device in target_devices:
            if device in self.knowledge_graph:
                # Get neighboring devices (same room)
                neighbors = list(self.knowledge_graph.neighbors(device))
                for neighbor in neighbors:
                    if self.knowledge_graph.nodes[neighbor].get("type") == "room":
                        # Get other devices in the same room
                        room_devices = [n for n in self.knowledge_graph.neighbors(neighbor) 
                                      if self.knowledge_graph.nodes[n].get("type") == "device"]
                        graph_features["affected_devices"].extend(room_devices)
        
        return graph_features
    
    def _identify_target_devices(self, command: str) -> List[str]:
        """Identify target devices mentioned in the command."""
        command_lower = command.lower()
        target_devices = []
        
        for node in self.knowledge_graph.nodes():
            node_data = self.knowledge_graph.nodes[node]
            if node_data.get("type") == "device":
                device_type = node_data.get("device_type", "").lower()
                if device_type in command_lower:
                    target_devices.append(node)
        
        return target_devices
    
    def _predict_device_states(self, command: str, current_time: datetime) -> Dict:
        """Predict device states after command execution."""
        device_states = {}
        
        # Get current states (default states for now)
        for node in self.knowledge_graph.nodes():
            node_data = self.knowledge_graph.nodes[node]
            if node_data.get("type") == "device":
                device_states[node] = {
                    "current_state": node_data.get("default_state", "unknown"),
                    "predicted_state": node_data.get("default_state", "unknown"),
                    "confidence": 0.8
                }
        
        # Update predictions based on command
        target_devices = self._identify_target_devices(command)
        for device in target_devices:
            if "turn on" in command.lower() or "start" in command.lower():
                device_states[device]["predicted_state"] = "on"
            elif "turn off" in command.lower() or "stop" in command.lower():
                device_states[device]["predicted_state"] = "off"
        
        return device_states
    
    def _build_spatial_context(self, command: str) -> Dict:
        """Build spatial context around the command."""
        target_devices = self._identify_target_devices(command)
        spatial_context = {
            "target_rooms": [],
            "adjacent_rooms": [],
            "affected_areas": []
        }
        
        for device in target_devices:
            if device in self.knowledge_graph:
                # Find the room containing this device
                for neighbor in self.knowledge_graph.neighbors(device):
                    if self.knowledge_graph.nodes[neighbor].get("type") == "room":
                        room = neighbor
                        spatial_context["target_rooms"].append(room)
                        
                        # Add adjacent rooms
                        if room in self.spatial_relationships:
                            spatial_context["adjacent_rooms"].extend(self.spatial_relationships[room])
        
        return spatial_context
    
    def _analyze_temporal_patterns(self, current_time: datetime) -> Dict:
        """Analyze temporal patterns and anomalies."""
        hour = current_time.hour
        
        return {
            "current_hour": hour,
            "time_category": self._categorize_time(hour),
            "is_unusual_time": self._is_unusual_time(current_time.time()),
            "sleep_time": self._is_sleep_time(current_time.time()),
            "peak_usage_time": 7 <= hour <= 9 or 17 <= hour <= 21
        }
    
    def _categorize_time(self, hour: int) -> str:
        """Categorize time of day."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _is_unusual_time(self, current_time: time) -> bool:
        """Check if current time is unusual for device operations."""
        return current_time < time(6, 0) or current_time > time(23, 0)
    
    def _is_sleep_time(self, current_time: time) -> bool:
        """Check if current time is typical sleep time."""
        return current_time >= time(22, 0) or current_time <= time(6, 0)
    
    def _extract_risk_indicators(self, command: str, context: Dict) -> Dict:
        """Extract risk indicators from command and context."""
        risk_indicators = {
            "high_risk_devices": [],
            "temporal_risk": False,
            "spatial_risk": False,
            "dependency_risk": False
        }
        
        # Check for high-risk devices
        high_risk_keywords = ["oven", "stove", "heater", "fireplace", "smoke_detector", "security"]
        for keyword in high_risk_keywords:
            if keyword in command.lower():
                risk_indicators["high_risk_devices"].append(keyword)
        
        # Check temporal risk
        if context.get("temporal_patterns", {}).get("is_unusual_time", False):
            risk_indicators["temporal_risk"] = True
        
        return risk_indicators
    
    def _analyze_device_dependencies(self, command: str) -> Dict:
        """Analyze device dependencies and cascading effects."""
        target_devices = self._identify_target_devices(command)
        dependencies = {
            "direct_dependencies": [],
            "indirect_dependencies": [],
            "potential_conflicts": []
        }
        
        # Simple dependency rules
        dependency_rules = {
            "oven": ["smoke_detector", "ventilation"],
            "security_camera": ["motion_sensor", "alarm_system"],
            "door_lock": ["security_system", "access_control"]
        }
        
        for device in target_devices:
            device_type = self.knowledge_graph.nodes.get(device, {}).get("device_type", "").lower()
            if device_type in dependency_rules:
                dependencies["direct_dependencies"].extend(dependency_rules[device_type])
        
        return dependencies
    
    def get_graph_statistics(self) -> Dict:
        """Get knowledge graph statistics."""
        if not self.knowledge_graph:
            return {"error": "No knowledge graph available"}
        
        return {
            "total_nodes": self.knowledge_graph.number_of_nodes(),
            "total_edges": self.knowledge_graph.number_of_edges(),
            "device_count": len([n for n in self.knowledge_graph.nodes() if self.knowledge_graph.nodes[n].get("type") == "device"]),
            "room_count": len([n for n in self.knowledge_graph.nodes() if self.knowledge_graph.nodes[n].get("type") == "room"]),
            "density": nx.density(self.knowledge_graph),
            "connected_components": nx.number_connected_components(self.knowledge_graph)
        }

if __name__ == "__main__":
    # Test the enhanced context builder
    print("Testing Enhanced Context Builder...")
    
    builder = EnhancedContextBuilder()
    
    print("\nKnowledge Graph Statistics:")
    stats = builder.get_graph_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTesting context building for sample commands...")
    
    test_commands = [
        "Turn off the smoke detector",
        "Set the oven to 500 degrees for 10 hours",
        "Turn on the TV in the living room",
        "Lock the front door"
    ]
    
    for command in test_commands:
        print(f"\nCommand: {command}")
        context = builder.build_enhanced_context(command)
        print(f"  Target devices: {context['knowledge_graph']['target_devices']}")
        print(f"  Risk indicators: {context['risk_indicators']['high_risk_devices']}")
        print(f"  Temporal risk: {context['risk_indicators']['temporal_risk']}")