from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import numpy as np
import dotenv, ocr, cv2

config = dotenv.dotenv_values(".env")
app = Flask(__name__)
app.secret_key = config["SECRET_KEY"]
CORS(app)

swagger_config = {
    "headers": [
    ],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}
swagger = Swagger(app, config=swagger_config)

@app.route(f"/api/status", methods=["GET"])
def status():
    """
    GET to check the status of the application
    ---
    responses:
      200:
        description: OK
        schema:
          id: Status
          properties:
            status:
              type: string
              description: The status of the application
              default: ok
    """
    return jsonify(status="OK")

@app.route(f"/api/scan", methods=["POST"])
def scan():
    """
    POST for image processing
    ---
    parameters:
      - name: img
        in: formData
        type: string
        format: binary
        description: Image to process
        required: true

    responses:
        200:
            description: A list of students absence
            schema:
                type: array
                items:
                    type: object
                    properties:
                        id:
                            type: string
                            description: The ID of the student
                        absence:
                            type: integer
                            description: The nuber of hour
        400:
            description: Bad or missing image

    """

    file = request.files.get("file")
    filename = file.filename.split(".")
    allowed_files = ["png", "jpg", "jpeg"]

    if file and filename[1] in allowed_files:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        try:
            data = ocr.Engine.process(image)
            return jsonify(data), 200
        except: None

    return jsonify({"error": "Bad or missing image"}), 400
    
if __name__ == '__main__':
    #app.run("0.0.0.0", 3001, debug=True)
    app.run("0.0.0.0", 3001)
