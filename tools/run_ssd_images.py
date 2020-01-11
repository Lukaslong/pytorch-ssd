import os
import sys
import glob
import cv2
import torch
import torchvision
import argparse

sys.path.insert(0,os.path.join(os.path.dirname(__file__), '..'))

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

def main(args):
    net_type = args.net_type
    img_folder=args.img_folder
    model_path = args.weights_path
    label_path = args.label_path
    class_names = [name.strip() for name in open(label_path).readlines()]
    out_path=args.out_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)

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

    img_names=glob.glob(img_folder+os.sep+'*.jpg')
    #result_csv=os.path.join(out_path,'rest_result.csv')
    if len(img_names)==0:
        print('No imagesfound in {}'.format(img_folder))
        exit(-1)

    for img_name in img_names:
        image=cv2.imread(img_name)

        timer.start()
        boxes,labels,probs=predictor.predict(image,10,0.3)
        interval = timer.end()

        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

        label_text=[]
        for i in range(boxes.size(0)):
            box=boxes[i,:]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            label_text.append(label)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(image, label,(box[0]+20, box[1]+40),cv2.FONT_HERSHEY_SIMPLEX,1,
                        (255, 0, 255), 2)

        if args.store_result:
            new_name='{}/{}'.format(out_path,img_name.split('/')[-1])
            cv2.imwrite(new_name,image)
            if not label_text:
                result_label='empty'
            else:
                result_label=label_text[0]
            with open(os.path.join(out_path,'rest_result.csv'),'a+') as result_writer:
                result_writer.write(img_name.split('/')[-1]+','+result_label+'\n')
                
        cv2.imshow('result', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Detection images in folder')
    parser.add_argument('--img-folder',type=str,help='path to image folder')
    parser.add_argument('--net-type',default='mb2-ssd-lite',type=str)
    parser.add_argument('--weights-path',help='path to trained weights')
    parser.add_argument('--label-path',help='path to label file')
    parser.add_argument('--out-path',help='path to save output images')
    parser.add_argument('--store-result',action='store_true',default=False)

    args=parser.parse_args()

    main(args)
