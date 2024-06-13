import numpy as np
import torch
import torchvision.transforms.functional as tvf
from Rapid import utils
from PIL import Image

class RapidDetector():
    '''
    Wrapper of image object detectors.

    Args:
        model_name: str, currently only support 'rapid'
        weights_path: str, path to the pre-trained network weights
        model: torch.nn.Module, used only during training
        conf_thres: float, confidence threshold
        input_size: int, input resolution
    '''
    def __init__(self, model_name='', weights_path=None, device=-1, 
        pd_input_resize=512, max_persons=1000, conf_thresh=0.02, nms_thresh=0.4):
        # assert torch.cuda.is_available()
        # cpu = device < 0  # boolean
        # self.device = torch.device("cpu" if cpu else "cuda")

        if device<0:
            self.device = torch.device("cpu")
        else: 
            self.device = torch.device("cuda:{}".format(device))
        # gpus = [device]

        # print("MAX DET PERSONS", max_persons)

        # post-processing settings
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.pd_input_resize = pd_input_resize

        self.top_K = max_persons
        self.keep_top_k = max_persons

        if model_name == 'rapid':
            # from Rapid.rapid import RAPiD
            # self.rapid_model = RAPiD(backbone='dark53')
            
            #modified Dec2021
            from Rapid.rapid_export import RAPiD
            self.rapid_model = RAPiD((self.pd_input_resize, self.pd_input_resize))
        else:
            raise NotImplementedError()

        print(f'Successfully initialized model {model_name}.')
        # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f'Successfully initialized model {model_name}.',
        #     'Total number of trainable parameters:', total_params)
        
        self.rapid_model.load_state_dict(torch.load(weights_path)['model'])
        print(f'Successfully loaded weights: {weights_path}')
        # self.rapid_model = self.rapid_model.to(self.device)

        # self.rapid_model = torch.nn.DataParallel(self.rapid_model, device_ids=[self.device]).cuda()
        self.rapid_model.eval()
        self.rapid_model = self.rapid_model.to(self.device)
        
        # torch.onnx.export(self.rapid_model, input, ONNX_FILE_PATH, input_names=['input'],
        #                   output_names=['output'], export_params=True)        

        # self.model = model.cuda()
        # self.top_K = 5000
        # self.keep_top_k = 750

    def predict_pil(self, pil_img):
        '''
        Args:
            pil_img: PIL.Image.Image
            input_size: int, input resolution
            conf_thres: float, confidence threshold
        '''
        # input_size = kwargs.get('input_size', self.input_size)
        # conf_thres = kwargs.get('conf_thres', self.conf_thres)
        # assert input_size is not None, 'Please specify the input resolution'
        # assert conf_thres is not None, 'Please specify the confidence threshold'

        assert isinstance(pil_img, Image.Image), 'input must be a PIL.Image'

        # pad to square
        input_size = self.pd_input_resize
        input_img, _, pad_info = utils.rect_to_square(pil_img, None, input_size, 0)
        
        input_ori = tvf.to_tensor(input_img)
        input_ = input_ori.unsqueeze(0)
        
        assert input_.dim() == 4
        input_ = input_.cuda(self.device)
        with torch.no_grad():
            dts = self.rapid_model(input_).cpu()

        dts = dts.squeeze()
        # post-processing
        dts = dts[dts[:,5] >= self.conf_thresh]

        if self.top_K>0 & (len(dts) > self.top_K):
            _, idx = torch.topk(dts[:,5], k=self.top_K)
            dts = dts[idx, :]

        # if kwargs.get('debug', False):
        #     np_img = np.array(input_img)
        #     visualization.draw_dt_on_np(np_img, dts)
        #     plt.imshow(np_img)
        #     plt.show()

        dts = utils.nms(dts, is_degree=True, nms_thres=self.nms_thresh, img_size=input_size)
        dts = utils.detection2original(dts, pad_info.squeeze())

        # if kwargs.get('debug', False):
        #     np_img = np.array(pil_img)
        #     visualization.draw_dt_on_np(np_img, dts)
        #     plt.imshow(np_img)
        #     plt.show()

        # print(dts)
        # np_img = np.array(pil_img)
        # visualization.draw_dt_on_np(np_img, dts)
        # return np_img
        
        return dts

    def predict_pil_batch(self, pil_imgs):
        '''
        Args:
            pil_img: PIL.Image.Image
            input_size: int, input resolution
            conf_thres: float, confidence threshold
        '''
        # input_size = kwargs.get('input_size', self.input_size)
        # conf_thres = kwargs.get('conf_thres', self.conf_thres)
        # assert input_size is not None, 'Please specify the input resolution'
        # assert conf_thres is not None, 'Please specify the confidence threshold'

        assert isinstance(pil_imgs[0], Image.Image), 'input must be a PIL.Image'

        # pad to square
        input_size = self.pd_input_resize
        batch_img = []
        for img in pil_imgs:
            input_img, _, pad_info = utils.rect_to_square(img, None, input_size, 0)
            input_ori = tvf.to_tensor(input_img)
            input_ = input_ori.unsqueeze(0)
            batch_img.extend(input_)
        
        # batch_img = torch.cat(batch_img, dim=0)
        batch_img = torch.stack(tuple(batch_img))

        assert batch_img.dim() == 4
        batch_img = batch_img.cuda(self.device)
        with torch.no_grad():
            dts = self.rapid_model(batch_img).cpu()

        detections = []
        for i in range(len(dts)):
            detection = dts[i].squeeze()
            detection = detection[detection[:,5] >= self.conf_thresh]
            if self.top_K>0 & (len(detection) > self.top_K):
                _, idx = torch.topk(detection[:,5], k=self.top_K)
                detection = detection[idx, :]

            detection = utils.nms(detection, is_degree=True, nms_thres=self.nms_thresh, img_size=input_size)
            detection = utils.detection2original(detection, pad_info.squeeze())
            detections.append(detection)

        # print("DTS size", len(dts), detections.shape)

        # # dts = dts.squeeze()
        # # post-processing
        # # dts = dts[dts[:,5] >= self.conf_thresh]

        # if self.top_K>0 & (len(dts) > self.top_K):
        #     _, idx = torch.topk(dts[:,5], k=self.top_K)
        #     dts = dts[idx, :]

        # if kwargs.get('debug', False):
        #     np_img = np.array(input_img)
        #     visualization.draw_dt_on_np(np_img, dts)
        #     plt.imshow(np_img)
        #     plt.show()

        # dts = utils.nms(dts, is_degree=True, nms_thres=self.nms_thresh, img_size=input_size)
        # dts = utils.detection2original(dts, pad_info.squeeze())

        # if kwargs.get('debug', False):
        #     np_img = np.array(pil_img)
        #     visualization.draw_dt_on_np(np_img, dts)
        #     plt.imshow(np_img)
        #     plt.show()

        # print(dts)
        # np_img = np.array(pil_img)
        # visualization.draw_dt_on_np(np_img, dts)
        # return np_img
        
        return detections

