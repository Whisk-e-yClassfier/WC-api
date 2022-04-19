from flask import Flask

from mmdet.apis import init_detector, inference_detector
import matplotlib

from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage
import datetime
from flask_cors import CORS
from S3Client import s3_connection

app = Flask(__name__)
CORS(app)
api = Api(app)
upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

config_file = './model/real_testconfig.py'
checkpoint_file = './model/latest_v4.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')
model.CLASSES = ('Aberfeldy 12',
                 'Aberlour 12',
                 'Ardbeg 10',
                 'Balvenie 12',
                 'Bowmore 12',
                 'Bushmills',
                 'Chivals Regal 12',
                 'Cragganmore 12',
                 'Famous grouse',
                 'Glen Grant 10',
                 'Glenfiddich 12',
                 'Glenfiddich 18',
                 'Glenlivet 12',
                 'Highland Park 12',
                 'J-B',
                 'Jim Beam',
                 'Johnnie walker black',
                 'Johnnie walker red',
                 'Laphroaig 10',
                 'Macallan 12',
                 'Maker-s Mark',
                 'Monkey shoulder',
                 'Nikka coffey',
                 'Singleton 12',
                 'Talisker 10',
                 'Wild Turkey 101',
                 'Woodford Reserve')

UPLOAD_DIR = "./files"
OUTPUT_DIR = "./files/out"
s3_client = s3_connection()


@api.route('/health')
class HealthCheck(Resource):
    def get(self):
        return {"status": "Healthy!"}


@api.route('/image')
@api.expect(upload_parser)
class HealthCheck(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded = args['file']
        filename = datetime.datetime.now().strftime("%Y_%d_%m__%H_%M_%S_%f.png")
        img_path = f'{UPLOAD_DIR}/{filename}'
        img_path_out = f'{OUTPUT_DIR}/{filename}'
        uploaded.save(img_path)

        result = inference_detector(model, img_path)
        matplotlib.pyplot.switch_backend('Agg')
        model.show_result(img_path, result, show=False,
                          score_thr=0.5, out_file=img_path_out)
        s3_client.upload_file(
            img_path_out, "your-bucket", f"wc/wc_{filename}")
        s3_url = f"your-s3-path/wc/wc_{filename}"
        return {"status": "success", "image_url": s3_url}  # response added
