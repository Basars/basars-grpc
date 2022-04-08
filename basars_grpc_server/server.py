import os
import logging

import grpc

from grpc import StatusCode
from concurrent import futures
from basars_grpc_server.models import create_stairs_vision_transformer
from basars_grpc_server.utils import grpc_postprocess, grpc_preprocess
from basars_grpc_core.protos.basars_pb2_grpc import BasarsServingServicer
from basars_grpc_core.protos.basars_pb2_grpc import add_BasarsServingServicer_to_server as add_service_to_server
from basars_grpc_core.protos.basars_pb2 import EndoscopicImageInput, PolypImageSlices


logging.basicConfig(level=logging.INFO)


class BasarsService(BasarsServingServicer):

    def __init__(self, model):
        self.model = model

    def provide(self, request: EndoscopicImageInput, context):
        pixels = grpc_preprocess(request.image, shape=(224, 224, 3))  # color inputs
        if pixels is None:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details('Input image must be shape=(224, 224, 3) or (150528,)')
            return PolypImageSlices()
        pixels = pixels / 255.
        logging.info('Performing endoscopic image analysis...')
        outputs = self.model.predict(pixels)[0]
        slices = [grpc_postprocess(outputs[:, :, axis]) for axis in range(outputs.shape[-1])]
        logging.info('Returning sliced result...')
        return PolypImageSlices(slices=slices)


def serve():
    logging.info('Loading Vision Transformer...')
    model = create_stairs_vision_transformer()
    service = BasarsService(model)

    grpc_host = os.getenv('BASARS_GRPC_HOST', '[::]')
    grpc_port = os.getenv('BASARS_GRPC_PORT', 9000)
    grpc_workers = os.getenv('BASARS_POOL_WORKERS', 10)
    grpc_workers = int(grpc_workers)

    logging.info('Setting gRPC server up...')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=grpc_workers))
    add_service_to_server(service, server)

    server.add_insecure_port('{}:{}'.format(grpc_host, grpc_port))
    server.start()
    logging.info('The gRPC server is listening on {}:{}'.format(grpc_host, grpc_port))

    server.wait_for_termination()


if __name__ == '__main__':
    serve()
