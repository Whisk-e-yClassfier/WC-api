import logging
from flask import Flask
from mmdet.apis import init_detector, inference_detector
import matplotlib

app = Flask(__name__)
if __name__ == "__main__":
    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    app.run(host="0.0.0.0")


@app.route("/")
def hello_world():
    # config = '../mmdetection/configs/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco.py'

    config_file = './model/testconfig_v2.py'
    # checkpoint_file = './model/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752-f22d2ce5.pth'
    checkpoint_file = './model/latest_v2.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu')
    img = './balvenie_test.jpg'
    result = inference_detector(model, img)
    matplotlib.pyplot.switch_backend('Agg')
    model.show_result(img, result, show=False, score_thr=0.000001,
                      out_file="balvenie_test_o.png")
    return {"data": "balvenie_test_o.png"}
