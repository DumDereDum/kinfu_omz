import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore


def getRGBlist(folder):
    f = open(folder + '\\rgb.txt', 'r')
    rgb = [folder + '\\' + s.replace('/', '\\') for s in f.read().split() if '.png' in s]
    return rgb


def midasnet_demo():
    # arguments
    parser = ArgumentParser()

    parser.add_argument(
        "-m", "--model", help="Required. Path to an .xml file with a trained model", required=True, type=Path)
    parser.add_argument(
        "-i", "--input", help="Required. Path to a input image file", required=True, type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="Optional. Required for CPU custom layers. Absolute MKLDNN (CPU)-targeted custom layers. "
                             "Absolute path to a shared library with the kernels implementations", type=str,
                        default=None)
    parser.add_argument("-d", "--device",
                        help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. "
                             "Sample will look for a suitable plugin for device specified. Default value is CPU",
                        default="CPU", type=str)

    args = parser.parse_args()

    # logging
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)

    log.info("creating inference engine")
    ie = IECore()
    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    log.info("Loading network")
    net = ie.read_network(args.model, args.model.with_suffix(".bin"))

    assert len(net.input_info) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    # read and pre-process input image
    _, _, height, width = net.input_info[input_blob].input_data.shape

    # loading model to the plugin
    log.info("loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
    rgb_list = getRGBlist(args.input)

    for s in rgb_list:
        print(s)
        image = cv2.imread(s, cv2.IMREAD_COLOR)
        cv2.imshow('input', image)
        (input_height, input_width) = image.shape[:-1]

        if (input_height, input_width) != (height, width):
            image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image_input = np.expand_dims(image, 0)
        res = exec_net.infer(inputs={input_blob: image_input})
        disp = np.squeeze(res[out_blob][0])
        disp = cv2.resize(disp, (input_width, input_height), cv2.INTER_CUBIC)

        disp_min = disp.min()
        disp_max = disp.max()

        if disp_max - disp_min > 1e-6:
            disp = (disp - disp_min) / (disp_max - disp_min)
        else:
            disp.fill(0.5)

        cv2.imshow('output', disp)
        cv2.waitKey(1)

if __name__ == '__main__':
    midasnet_demo()
