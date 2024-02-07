import pickle

dayofweek_dict = {
    'day:Mon': 0,
    'day:Tue': 1,
    'day:Wed': 2,
    'day:Thu': 3,
    'day:Fri': 4,
    'day:Sat': 5,
    'day:Sun': 6,
}

hour_dict = {
    'time:(0~3)': 0,
    'time:(3~6)': 1,
    'time:(6~9)': 2,
    'time:(9~12)': 3,
    'time:(12~15)': 4,
    'time:(15~18)': 5,
    'time:(18~21)': 6,
    'time:(21~24)': 7,
}

device_dict = {
    'AirConditioner': 0,  # 空气调节器
    'AirPurifier': 1,  # 空气净化器
    'Blind': 2,  # 窗帘
    'Camera': 3,  # 摄像机
    'ClothingCareMachine': 4,  # 衣物护理机
    'Computer': 5,  # 电脑
    'ContactSensor': 6,  # 接触传感器（lock）
    'CurbPowerMeter': 7,  # 抑制功率计
    'Dishwasher': 8,  # 洗碗机
    'Dryer': 9,  # 干燥机
    'Elevator': 10,  # 电梯
    'Fan': 11,  # 风扇
    'GarageDoor': 12,  # 车库门
    'Light': 13,  # 灯
    'Microwave': 14,  # 微波炉
    'MotionSensor': 15,  # 运动传感器
    'NetworkAudio': 16,  # 运动传感器
    'None': 17,  # None
    'Other': 18,
    'Oven': 19,  # 烤箱
    'PresenceSensor': 20,  # 存在传感器
    'Projector': 21,  # 投影仪
    'Refrigerator': 22,  # 冰箱
    'RemoteController': 23,  # 遥控器
    'RobotCleaner': 24,  # 机器人清洁工
    'Siren': 25,  # 警报器
    'SmartLock': 26,  # 智能锁
    'SmartPlug': 27,  # 智能插头
    'Switch': 28,  # 开关
    'Television': 29,  # 电视
    'Thermostat': 30,  # 恒温器
    'Washer': 31,  # 洗衣机
    'WaterValve': 32  # 水阀
}

device_control_dict = {
    'AirConditioner:fanspeedDown': 0,
    'AirConditioner:fanspeedUp': 1,
    'AirConditioner:notification': 2,
    'AirConditioner:refresh refresh': 3,
    'AirConditioner:setAcOptionalMode': 4,
    'AirConditioner:setAirConditionerMode': 5,
    'AirConditioner:setCoolingSetpoint': 6,
    'AirConditioner:setFanMode': 7,
    'AirConditioner:setOutingMode': 8,
    'AirConditioner:switch off': 9,
    'AirConditioner:switch on': 10,
    'AirConditioner:switch toggle': 11,
    'AirConditioner:temperatureDown': 12,
    'AirConditioner:temperatureUp': 13,
    'AirPurifier:notification': 14,
    'AirPurifier:refresh refresh': 15,
    'AirPurifier:setAirPurifierMode': 16,
    'AirPurifier:setFanMode': 17,
    'AirPurifier:setFanSpeed': 18,
    'AirPurifier:switch off': 19,
    'AirPurifier:switch on': 20,
    'AirPurifier:turnWelcomeCareOn': 21,
    'Blind:refresh refresh': 22,
    'Blind:statelessCurtainPowerButton setButton': 23,
    'Blind:switch off': 24,
    'Blind:switch on': 25,
    'Blind:switchLevel setLevel': 26,
    'Blind:windowShade close': 27,
    'Blind:windowShade open': 28,
    'Blind:windowShadeLevel setShadeLevel': 29,
    'Blind:windowShadePreset presetPosition': 30,
    'Camera:alarm off': 31,
    'Camera:cameraPreset execute': 32,
    'Camera:imageCapture take': 33,
    'Camera:notification': 34,
    'Camera:refresh refresh': 35,
    'Camera:switch off': 36,
    'Camera:switch on': 37,
    'Camera:videoCapture capture': 38,
    'ClothingCareMachine:dryerOperatingState setMachineState stop': 39,
    'ClothingCareMachine:notification': 40,
    'ClothingCareMachine:refresh refresh': 41,
    'Computer:refresh refresh': 42,
    'ContactSensor:doorControl close': 43,
    'ContactSensor:lock lock': 44,
    'ContactSensor:lock unlock': 45,
    'ContactSensor:refresh refresh': 46,
    'ContactSensor:switch off': 47,
    'ContactSensor:switch on': 48,
    'ContactSensor:switch toggle': 49,
    'CurbPowerMeter:energyMeter resetEnergyMeter': 50,
    'Dishwasher:notification': 51,
    'Dishwasher:refresh refresh': 52,
    'Dishwasher:start': 53,
    'Dryer:dryerOperatingState setMachineState pause': 54,
    'Dryer:dryerOperatingState setMachineState run': 55,
    'Dryer:notification': 56,
    'Dryer:refresh refresh': 57,
    'Dryer:switch on': 58,
    'Elevator:elevatorCall call': 59,
    'Elevator:refresh refresh': 60,
    'Fan:fanSpeed setFanSpeed': 61,
    'Fan:notification': 62,
    'Fan:refresh refresh': 63,
    'Fan:switch off': 64,
    'Fan:switch on': 65,
    'GarageDoor:doorControl close': 66,
    'GarageDoor:doorControl open': 67,
    'GarageDoor:notification': 68,
    'GarageDoor:refresh refresh': 69,
    'GarageDoor:switch off': 70,
    'GarageDoor:switch on': 71,
    'Light:refresh refresh': 72,
    'Light:setColor': 73,
    'Light:setColorTemperature': 74,
    'Light:setLevel': 75,
    'Light:setLightingMode': 76,
    'Light:switch off': 77,
    'Light:switch on': 78,
    'Light:switch toggle': 79,
    'Microwave:refresh refresh': 80,
    'Microwave:switch on': 81,
    'MotionSensor:circlemusic21301.motionCommands active': 82,
    'NetworkAudio:audioMute mute': 83,
    'NetworkAudio:audioVolume setVolume': 84,
    'NetworkAudio:mediaPlayback pause': 85,
    'NetworkAudio:mediaPlayback play': 86,
    'NetworkAudio:mediaPlayback stop': 87,
    'NetworkAudio:mediaTrackControl nextTrack': 88,
    'NetworkAudio:mute': 89,
    'NetworkAudio:notification': 90,
    'NetworkAudio:refresh refresh': 91,
    'NetworkAudio:setVolume': 92,
    'NetworkAudio:switch off': 93,
    'NetworkAudio:switch on': 94,
    'NetworkAudio:unmute': 95,
    'None:location': 96,
    'None:scene': 97,
    'None:sleep': 98,
    'Other:audioMute mute': 99,
    'Other:audioMute unmute': 100,
    'Other:colorControl setColor': 101,
    'Other:colorTemperature setColorTemperature': 102,
    'Other:custom.picturemode setPictureMode': 103,
    'Other:custom.soundmode setSoundMode': 104,
    'Other:eventstreet19532.chmode setMode': 105,
    'Other:notification': 106,
    'Other:notification deviceNotification': 107,
    'Other:perfectpiano41701.boundaryIntruderAlarm setSecuritySystem': 108,
    'Other:refresh refresh': 109,
    'Other:robotCleanerTurboMode setRobotCleanerTurboMode': 110,
    'Other:samsungvd.mediaInputSource setInputSource': 111,
    'Other:signalahead13665.coffeemakerprogramsv2 setProgram': 112,
    'Other:signalahead13665.startstopprogramv2 setStartstop': 113,
    'Other:switch off': 114,
    'Other:switch on': 115,
    'Other:switch toggle': 116,
    'Other:switchLevel setLevel': 117,
    'Other:thermostatCoolingSetpoint setCoolingSetpoint': 118,
    'Other:windowShade close': 119,
    'Other:windowShade open': 120,
    'Other:windowShadeLevel setShadeLevel': 121,
    'Oven:signalahead13665.ovenprogramsv2 setProgram': 122,
    'Oven:signalahead13665.pauseresumev2 setPauseState': 123,
    'Oven:signalahead13665.programdurationv2 setProgramDuration': 124,
    'Oven:signalahead13665.startstopprogramv2 setStartstop': 125,
    'Oven:switch off': 126,
    'Oven:switch on': 127,
    'PresenceSensor:switch off': 128,
    'PresenceSensor:switch on': 129,
    'Projector:audioMute mute': 130,
    'Projector:custom.picturemode setPictureMode': 131,
    'Projector:custom.soundmode setSoundMode': 132,
    'Projector:samsungvd.ambient setAmbientOn': 133,
    'Projector:samsungvd.mediaInputSource setInputSource': 134,
    'Projector:switch off': 135,
    'Projector:switch on': 136,
    'Refrigerator:notification': 137,
    'Refrigerator:refresh refresh': 138,
    'Refrigerator:samsungce.powerCool activate': 139,
    'Refrigerator:samsungce.powerCool deactivate': 140,
    'Refrigerator:samsungce.powerFreeze activate': 141,
    'Refrigerator:samsungce.powerFreeze deactivate': 142,
    'Refrigerator:setCoolTemperature': 143,
    'Refrigerator:switch off': 144,
    'Refrigerator:switch on': 145,
    'RemoteController:momentary push': 146,
    'RemoteController:switch off': 147,
    'RemoteController:switch on': 148,
    'RobotCleaner:notification': 149,
    'RobotCleaner:refresh refresh': 150,
    'RobotCleaner:setRobotCleanerCleaningMode auto': 151,
    'RobotCleaner:setRobotCleanerCleaningMode map': 152,
    'RobotCleaner:setRobotCleanerCleaningMode part': 153,
    'RobotCleaner:setRobotCleanerMovement after': 154,
    'RobotCleaner:setRobotCleanerMovement charging': 155,
    'RobotCleaner:setRobotCleanerMovement cleaning': 156,
    'RobotCleaner:setRobotCleanerMovement homing': 157,
    'RobotCleaner:setRobotCleanerMovement point': 158,
    'RobotCleaner:setRobotCleanerMovement powerOff': 159,
    'RobotCleaner:setRobotCleanerMovement reserve': 160,
    'RobotCleaner:setRobotCleanerTurboMode off': 161,
    'RobotCleaner:setRobotCleanerTurboMode on': 162,
    'RobotCleaner:setRobotCleanerTurboMode silence': 163,
    'RobotCleaner:setting': 164,
    'Siren:alarm both': 165,
    'Siren:alarm off': 166,
    'Siren:notification': 167,
    'SmartLock:alarm both': 168,
    'SmartLock:alarm off': 169,
    'SmartLock:lock lock': 170,
    'SmartLock:lock unlock': 171,
    'SmartLock:refresh refresh': 172,
    'SmartLock:switch off': 173,
    'SmartLock:switch on': 174,
    'SmartPlug:powerToggle': 175,
    'SmartPlug:refresh refresh': 176,
    'SmartPlug:setting': 177,
    'SmartPlug:switch off': 178,
    'SmartPlug:switch on': 179,
    'Switch:refresh refresh': 180,
    'Switch:setColor': 181,
    'Switch:setColorTemperature': 182,
    'Switch:setCoolingSetpoint': 183,
    'Switch:setLevel': 184,
    'Switch:setThermostatMode': 185,
    'Switch:switch off': 186,
    'Switch:switch on': 187,
    'Switch:switch toggle': 188,
    'Switch:windowShade close': 189,
    'Switch:windowShade open': 190,
    'Television:audioMute mute': 191,
    'Television:audioMute unmute': 192,
    'Television:refresh refresh': 193,
    'Television:setAmbientContent': 194,
    'Television:setAmbientOn': 195,
    'Television:setChannel': 196,
    'Television:setInputSource': 197,
    'Television:setPictureMode': 198,
    'Television:setSoundMode': 199,
    'Television:setVolume': 200,
    'Television:switch off': 201,
    'Television:switch on': 202,
    'Television:switch toggle': 203,
    'Television:textMessage': 204,
    'Television:volumeDown': 205,
    'Thermostat:notification': 206,
    'Thermostat:refresh refresh': 207,
    'Thermostat:setCoolingSetpoint': 208,
    'Thermostat:setHeatingSetpoint': 209,
    'Thermostat:setMode': 210,
    'Thermostat:setThermostatFanMode': 211,
    'Thermostat:setThermostatMode': 212,
    'Thermostat:switch off': 213,
    'Thermostat:switch on': 214,
    'Washer:act': 215,
    'Washer:refresh refresh': 216,
    'Washer:washerOperatingState setMachineState pause': 217,
    'Washer:washerOperatingState setMachineState run': 218,
    'Washer:washerOperatingState setMachineState stop': 219,
    'WaterValve:valve close': 220,
    'WaterValve:valve open': 221
}


def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


if __name__ == "__main__":
    data = read_pkl_file("test_instance_10.pkl")
    for item in data:
        for seq in item:
            for s in seq:
                if s == 228:
                    print("err")
            # if i == 228:
            #     print("err")
