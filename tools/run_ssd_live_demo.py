import cv2
import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

def main(args):
    net_type = args.net_type
    model_path = args.weights_path
    label_path = args.label_path
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)

    if args.live:
        cap = cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
    else:
        cap=cv2.VideoCapture(args.video_path)
    
    out_video = args.out_video
    Fourcc=cv2.VideoWriter_fourcc('M','P','4','V')
    writer=cv2.VideoWriter(out_video,fourcc=Fourcc,fps=15,frameSize=(640,480))

    num_gpus=torch.cuda.device_count()
    device='cuda' if num_gpus else 'cpu'

    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    #elif net_type == 'mb3-ssd-lite':
    #    net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)

    net.load(model_path)

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=20,device=device)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=20,device=device)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=20,device=device)
    elif net_type == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=20,device=device)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=20,device=device)
    #elif net_type == 'mb3-ssd-lite':
    #    predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=10)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)


    timer = Timer()

    while True:
        _, orig_image = cap.read()
        if orig_image is None:
            print('END')
            break

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        timer.start()
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        interval = timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        writer.write(orig_image)
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Live Detection')
    parser.add_argument('--net-type',default='mb3-ssd-lite',choices=['mb2-ssd-lite','mb3-ssd-lite'],type=str)
    parser.add_argument('--weights-path',help='path to trained weights')
    parser.add_argument('--label-path',help='path to label file')
    parser.add_argument('--live',action='store_true',default=False)
    parser.add_argument('--video-path',help='path to video')
    parser.add_argument('--out-video',help='output video names')

    args=parser.parse_args()

    main(args)
