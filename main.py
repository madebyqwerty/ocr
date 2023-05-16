from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from flasgger import Swagger
import dotenv

VERSION = "v0"
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
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}
swagger = Swagger(app, config=swagger_config)


@app.route(f"/api/{VERSION}/status", methods=["GET"])
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

@app.route(f"/api/{VERSION}/process_img", methods=["POST"])
def process_img():
    """
    POST for image processing
    ---
    parameters:
      - name: img
        in: formData
        type: img
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
                            type: integer
                            description: The ID of the student
                        absence:
                            type: integer
                            description: The nuber of hour
        400:
            description: Bad input image

    """
    return jsonify([{"id": 8347, "absence": 0}, {"id": 8347, "absence": 1}, {"id": 8347, "absence": 3}])
    
if __name__ == '__main__':
    #app.run("localhost", 5000, debug=True)
    app.run("localhost", 5000)
