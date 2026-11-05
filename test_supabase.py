
from services.cloud_database import test_connection

try:
    devices = test_connection()
    print("Supabase connection successful!")
    print("Devices:", devices)
except Exception as error:
    print("Supabase connection failed.")
    print(type(error).__name__, str(error))
