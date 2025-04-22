from http.server import BaseHTTPRequestHandler
import json
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import base64
import io

model = YOLO("best.pt")

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        
        results = model.predict(img)
        boxes = results[0].boxes.data
        class_names = [model.names[int(cls)] for cls in boxes[:, 5]]

        count = Counter(class_names)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "WBC": count.get("WBC", 0),
            "RBC": count.get("RBC", 0),
            "Platelets": count.get("platelets", 0)
        }
        
        self.wfile.write(json.dumps(response).encode())