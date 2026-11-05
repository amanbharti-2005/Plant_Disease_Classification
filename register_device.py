from services.cloud_database import register_device

try:
    device = register_device()
    print("Rover registered successfully!")
    print("Device ID:", device["id"])
    print("Device name:", device["device_name"])
except Exception as error:
    print("Registration failed:", type(error).__name__, str(error))
    
