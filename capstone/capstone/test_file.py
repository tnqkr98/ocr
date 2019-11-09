import main
import fire
import pprint
import werkzeug
from waitress import serve
from flask import Flask, request
from flask_json import json_response, as_json
from flask_cors import CORS
from cachetools import cached, TTLCache



"""
"""
import pandas as pd
from file_util import load_files, crop_img

import text_recognition.demo as recognition
import text_detection.test as detection

# Do not use the development server in a production environment.
# Create the application instance
app = Flask(__name__)
CORS(app)
app.config.from_object(__name__)  # Load config from app.py file
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['MAX_CONTENT_LENGTH'] = (1024 * 1024) * 25  # 5MB
app.config['ENV_DEFAULT_PORT'] = "8000"
app.config['ENV_DEBUG_MODE'] = True
app.config['JSON_ADD_STATUS'] = True
app.config['JSON_STATUS_FIELD_NAME'] = 'server_status'
app.config['JSON_JSONP_OPTIONAL'] = False
app.config['JSON_DECODE_ERROR_MESSAGE'] = True
app.config['Threaded'] = True
app.config['APP_HOST_NAME'] = '127.0.0.1'

cache = TTLCache(maxsize=300, ttl=360)


def response_json_ops(custom_status=200, status=200, res_msg='Put on a happy face'):
    return json_response(server_status=custom_status, status_=status, message=res_msg)


def server_ops():
    p = pprint.PrettyPrinter(indent='4')
    # ------
    # Display server information :)
    p.pprint(app.config)
    # To allow aptana to receive errors, set use_debugger=False
    # app.run(port=app.config['ENV_DEFAULT_PORT'], debug=app.config['ENV_DEBUG_MODE'])
    # Deploy Server with Web Serve Gateway Interface
    serve(app=app, host=app.config['APP_HOST_NAME'], port=app.config['ENV_DEFAULT_PORT'])


@app.route('/', methods=["GET"])
@as_json
def test_api_get():
    response_header = response_json_ops()

    return response_header


@cached(cache)
def read_data():
    main.run_main()


@app.route('/', methods=["POST"])
@as_json
def test_api_post():
    response_header = response_json_ops()
    # ------
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save("./text_detection/test/" + filename)
    detection.run_detection()

    img_files, img_bbox = load_files()
    crop_img(img_files, img_bbox)
    pred_str = recognition.run_recognition()

    # [l, t], [r, t], [r, b], [l, b]
    for i, file in enumerate(img_files):
        txt = pd.read_csv(img_bbox[i], header=None)
        df = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "result_text"])

        for num, _col in enumerate(list(df)[:-1]):
            df[_col] = txt[num]
        df["result_text"] = pred_str
        df.to_csv("./result.csv")
    return response_header


if __name__ == '__main__':
    fire.Fire(server_ops)
