import hashlib
import os
import base64


def generate_random_id(length=32):
    random_bytes = os.urandom(16)    
    sha256 = hashlib.sha256(random_bytes)
    random_id = sha256.hexdigest()
    return random_id[:length]

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')