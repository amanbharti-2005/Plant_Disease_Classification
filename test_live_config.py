
import config

print("Live monitoring:", config.ENABLE_LIVE_MONITORING)
print("Stream URL:", config.LIVE_STREAM_URL)
print("Inference interval:", config.INFERENCE_INTERVAL_SECONDS)
print(
    "Negative observation interval:",
    config.NEGATIVE_OBSERVATION_INTERVAL_SECONDS
)
print("Cloud persistence:", config.ENABLE_SUPABASE_PERSISTENCE)
print("Image bucket:", config.SUPABASE_IMAGE_BUCKET)
print("Weed model enabled:", config.ENABLE_WEED_DETECTION)
