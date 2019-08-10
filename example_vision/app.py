# coding=utf-8
# Created by Meteorix at 2019/8/9
from flask import Flask, jsonify, request
from gevent import monkey

from service_streamer import ThreadedStreamer
from .model import batch_predict

monkey.patch_all()
app = Flask(__name__)
streamer = ThreadedStreamer(batch_predict, batch_size=64)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = streamer.predict([img_bytes])[0]
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == "__main__":
    from gevent.pywsgi import WSGIServer

    WSGIServer(("0.0.0.0", 5005), app).serve_forever()
