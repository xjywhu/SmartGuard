import json

def load_home_system(file_path="commandData/homeSystem.json"):
    """Loads the home system definition from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            home_system = json.load(f)
        print(f"Successfully loaded home system from {file_path}")
        return home_system
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return None

def load_commands(file_path="commandData/120datasetmixJson.json"):
    """Loads the commands from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            commands = json.load(f)
        print(f"Successfully loaded commands from {file_path}")
        return commands
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return None

if __name__ == "__main__":
    print("Starting KG Builder and Command Processor...")
    
    # Construct absolute paths relative to the script's assumed location in d:\SmartGuard
    # This assumes the script is run from d:\SmartGuard
    # For robustness, consider using os.path.join and os.path.dirname(__file__)
    # but for now, we'll keep it simple as per the current project structure.
    
    # Construct absolute paths
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__)) # d:\SmartGuard
    home_system_path = os.path.join(script_dir, "commandData", "homeSystem.json")
    commands_path = os.path.join(script_dir, "commandData", "120datasetmixJson.json")
    
    print(f"DEBUG: Attempting to load home system data from: {home_system_path}") # Added debug print
    home_system_data = load_home_system(home_system_path)
    if not home_system_data:
        print(f"DEBUG: Failed to load home system data from {home_system_path}. Check file existence and JSON format.")

    print(f"DEBUG: Attempting to load commands data from: {commands_path}") # Added debug print
    commands_data = load_commands(commands_path)
    if not commands_data:
        print(f"DEBUG: Failed to load commands data from {commands_path}. Check file existence and JSON format.")
    
    if home_system_data and "Home_System" in home_system_data:
        actual_system_data = home_system_data["Home_System"]
        # Rooms are direct keys under "Home_System", not under a "Rooms" sub-key
        # We can count rooms by counting keys in actual_system_data that are not "Whole_House_Systems"
        room_count = sum(1 for key in actual_system_data if key != "Whole_House_Systems")
        print(f"Loaded {room_count} rooms from home system data.")
        # print(actual_system_data) # Uncomment to see the loaded structure
    else:
        print("Home system data could not be loaded or 'Home_System' key is missing. Cannot proceed.")
        actual_system_data = {} # Ensure it's an empty dict to prevent further errors

    print("\n--- Processing Commands ---")
    # Use actual_system_data for command processing
    if actual_system_data and commands_data:
        for i, command_entry in enumerate(commands_data):
            action = command_entry.get("action")
            # Ensure device_name_from_command is fetched here
            device_name_from_command = command_entry.get("device") 
            parameters = command_entry.get("parameters")
            location_from_command = command_entry.get("location")
            
            # Determine if this is the target command for detailed debugging EARLIER
            # but after normalization of location
            temp_normalized_location_for_target_check = location_from_command
            if location_from_command and location_from_command.lower() == "study":
                temp_normalized_location_for_target_check = "Study_Office"
            elif location_from_command and location_from_command.lower() == "living room":
                temp_normalized_location_for_target_check = "Living_Room"
            
            is_target_command_early = device_name_from_command and device_name_from_command.lower() == "computer" and \
                                  location_from_command and location_from_command.lower() == "study"

            if is_target_command_early:
                print(f"\nProcessing TARGET Command {i+1}: {command_entry}")
                # print(f"DEBUG (Target Pre-Normalization): Device='{device_name_from_command}', Location='{location_from_command}'")
            # else:
                # General print removed to reduce verbosity
                # print(f"\nProcessing Command {i+1}: {command_entry}")

            if not all([action, device_name_from_command, location_from_command]):
                # print(f"Skipping command due to missing action, device, or location: {command_entry}")
                continue

            # Normalize location names (main normalization for logic)
            normalized_location = location_from_command
            if location_from_command.lower() == "study":
                normalized_location = "Study_Office"
            elif location_from_command.lower() == "living room":
                normalized_location = "Living_Room"
            # Add more mappings as needed

            device_found = False
            device_details = None
            matched_kg_device_name = device_name_from_command # Default

            # Re-evaluate is_target_command after full normalization for use in deeper logic
            is_target_command = device_name_from_command.lower() == "computer" and normalized_location == "Study_Office"

            # Attempt to find the device
            if is_target_command:
                # Use actual_system_data here
                available_room_keys = [key for key in actual_system_data if key != "Whole_House_Systems"]
                print(f"DEBUG (Target): About to check location. normalized_location='{normalized_location}'. Available rooms: {available_room_keys}")

            if normalized_location.lower() == "whole house":
                # Use actual_system_data here
                if "Whole_House_Systems" in actual_system_data:
                    whole_house_devices = actual_system_data["Whole_House_Systems"]
                if "Whole_House_Systems" in home_system_data:
                    whole_house_devices = home_system_data["Whole_House_Systems"]
                    if device_name_from_command in whole_house_devices:
                        device_details = whole_house_devices[device_name_from_command]
                        device_found = True
                    else:
                        for kg_device_name, details_wh in whole_house_devices.items():
                            if device_name_from_command.lower() in kg_device_name.lower().replace("_", " ") or \
                               kg_device_name.lower().replace("_", " ") in device_name_from_command.lower():
                                device_details = details_wh
                                matched_kg_device_name = kg_device_name
                                device_found = True
                                original_command_device_name_wh = command_entry.get("device")
                                print(f"Matched command device '{original_command_device_name_wh}' to Whole House System '{kg_device_name}'")
                                break
            # Use actual_system_data here
            elif normalized_location in actual_system_data: # Check directly in actual_system_data for room names
                room_devices = actual_system_data[normalized_location]
                if device_name_from_command in room_devices: # Direct case-sensitive match
                    device_details = room_devices[device_name_from_command]
                    matched_kg_device_name = device_name_from_command # KG name is same as command name
                    device_found = True
                    print(f"Matched command device '{device_name_from_command}' to KG device '{matched_kg_device_name}' in '{normalized_location}' (direct match)")
                else: # Direct case-sensitive key match failed, try substring
                    if is_target_command:
                        print(f"DEBUG (Target): Direct key match failed for '{device_name_from_command}' in '{normalized_location}'. Trying substring match.")
                        print(f"DEBUG (Target): Iterating through devices in '{normalized_location}': {list(room_devices.keys())}")

                    for kg_name_iter, kg_details_iter in room_devices.items():
                        cmd_name_lower = device_name_from_command.lower()
                        kg_name_iter_lower_norm = kg_name_iter.lower().replace("_", " ")
                        
                        if is_target_command:
                            print(f"DEBUG (Target): Comparing CMD:'{cmd_name_lower}' with KG_NORM:'{kg_name_iter_lower_norm}' (from KG key '{kg_name_iter}')")
                        
                        match_cond1 = cmd_name_lower in kg_name_iter_lower_norm
                        match_cond2 = kg_name_iter_lower_norm in cmd_name_lower
                        
                        if is_target_command:
                            print(f"DEBUG (Target): cond1 ('{cmd_name_lower}' in '{kg_name_iter_lower_norm}') = {match_cond1}")
                            print(f"DEBUG (Target): cond2 ('{kg_name_iter_lower_norm}' in '{cmd_name_lower}') = {match_cond2}")

                        if match_cond1 or match_cond2:
                            if is_target_command:
                                print(f"DEBUG (Target): Substring match SUCCESS: CMD:'{cmd_name_lower}', KG_NORM:'{kg_name_iter_lower_norm}'")
                            device_details = kg_details_iter
                            matched_kg_device_name = kg_name_iter # Use the actual KG key name
                            device_found = True
                            original_command_device_name_room = command_entry.get("device") 
                            print(f"Matched command device '{original_command_device_name_room}' to KG device '{matched_kg_device_name}' in '{normalized_location}' (substring match)")
                            break # Exit this inner for loop (found a match)
                        else:
                            if is_target_command:
                                print(f"DEBUG (Target): Substring match FAILED: CMD:'{cmd_name_lower}', KG_NORM:'{kg_name_iter_lower_norm}'")
                    
                    if not device_found and is_target_command:
                        print(f"DEBUG (Target): Substring search completed for '{device_name_from_command}' in {normalized_location}, no match found in loop.")

            # After attempting to find the device
            if device_found:
                # Construct the 'action to be taken' string
                action_to_be_taken_str = f"{action}"
                if parameters:
                    param_str_parts = []
                    for p_key, p_val in parameters.items():
                        param_str_parts.append(f"{p_key.replace('_', ' ')}: {p_val}")
                    if param_str_parts:
                        action_to_be_taken_str += f" ({', '.join(param_str_parts)})"
                
                parsed_action = {
                    "area_of_action": normalized_location,
                    "device": matched_kg_device_name, # Use the name from the KG
                    "action_to_be_taken": action_to_be_taken_str
                }
                print(f"Structured Action: {parsed_action}")
                # Here you would typically pass parsed_action to the next stage (e.g., risk assessment)
            else:
                if is_target_command or not (normalized_location.lower() == "whole house" or normalized_location in actual_system_data):
                    # Only print 'not found' for target or if location itself was bad, to reduce noise
                    print(f"Device '{device_name_from_command}' in location '{location_from_command}' (normalized to '{normalized_location}') not found or details missing.")

    else:
        print("Could not load home system data or commands data. Processing aborted.")

    print("\n--- KG Builder and Command Processor Finished ---")