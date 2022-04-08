import os
import grpc
import cv2

import numpy as np

from basars_grpc_core.protos.basars_pb2_grpc import BasarsServingStub
from basars_grpc_core.protos.basars_pb2 import EndoscopicImageInput
from basars_grpc_server.utils import grpc_postprocess, grpc_preprocess
from basars_grpc_client.postprocessing import save_as_readable_image


def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img


def run():
    grpc_host = os.getenv('BASARS_HOST', 'localhost')
    grpc_port = os.getenv('BASARS_PORT', 9000)

    source_images_dir = os.getenv('BASARS_IMAGE_SOURCE_DIR', 'sample_images')
    target_images_dir = os.getenv('BASARS_IMAGE_TARGET_DIR', 'target_images')

    print('Joining in to the gRPC server')
    channel = grpc.insecure_channel('{}:{}'.format(grpc_host, grpc_port))
    stub = BasarsServingStub(channel)
    print('Successfully connected to the gRPC server: {}/{}'.format(grpc_host, grpc_port))

    for filename in os.listdir(source_images_dir):
        filepath = os.path.join(source_images_dir, filename)
        print('Committing image to gRPC server: {}'.format(filepath))
        sample_img = load_image(filepath)
        img = grpc_postprocess(sample_img)
        response = stub.provide(EndoscopicImageInput(image=img))
        print('Responded from server: {}'.format(filepath))

        phase_images = []
        for buffer in response.slices:
            pixels = grpc_preprocess(buffer, dtype=np.float32, shape=(224, 224, 1))
            phase_images.append(pixels[0])  # squeeze the first dimension

        name_only = filename.split('.')[0]
        if not os.path.exists(target_images_dir):
            os.makedirs(target_images_dir, exist_ok=True)

        dst_filepath = '{}/analysis_{}.jpg'.format(target_images_dir, name_only)
        save_as_readable_image(sample_img, phase_images, dst_filepath)
        print('The readable image have been saved at: {}'.format(dst_filepath))


if __name__ == '__main__':
    run()
