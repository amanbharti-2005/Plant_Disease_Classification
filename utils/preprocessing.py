import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_image(path):
    # 1. Load image (Keras handles RGB natively)
    img = tf.keras.utils.load_img(
        path,
        target_size=IMG_SIZE
    )
    
    # 2. Convert to numpy array
    img = tf.keras.utils.img_to_array(img)
    
    # 3. Add batch dimension -> shapes it to (1, 224, 224, 3)
    # CRITICAL: DO NOT DIVIDE BY 255.0 AND DO NOT USE preprocess_input HERE!
    img = np.expand_dims(img, axis=0)
    
    return img