import os
# import base64

import cv2
import numpy as np
from skimage import transform
# from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# from Engines.rapid_detector import RapidDetector
# from Pose.detect import HumanPoseDetection
# from Engines.embedding import UbodyEmbedding

# from Engines.rapid_detector_16fp import RapidDetector
# from Pose.detect_16fp import HumanPoseDetection
from Pose.detect import HumanPoseDetection
# from Engines.embedding import UbodyEmbedding, FullbodyEmbedding
from Engines.embedding_16fp import UbodyEmbedding
from Engines.person_tracker import FishEyeTracker
from Engines import visualization


from common.database import Database

from collections import OrderedDict
from scipy.spatial import distance as dist

from PIL import Image
import time
import math
import pandas as pd
from termcolor import colored
from queue import Queue

import tensorrt as trt
import torch
import torchvision.transforms.functional as tvf
from Rapid import utils
import pycuda.autoinit
import pycuda.driver as pcuda 

class PersonEngine(object):
    def __init__(self, pd_model, pr_model, kp_model, kp_cfg, 
        body_db_file, pd_det_threshold=0.5, pd_nms_threshold=0.3, pd_input_resize=0, 
        max_detected_persons=0, min_person_width=50, 
        pr_threshold=0.3, device=-1, fisheye=0):

        # self.rgb = rgb
        self.min_person_width = min_person_width
        self.frame_num = 0
        self.fisheye = fisheye
        self.LMK_VISIBILITY_THRESHOLD = 0

        self.pr_threshold = pr_threshold

        self.detection_engine = DetectionEngine(pd_model, pd_det_threshold, pd_nms_threshold,
                                pd_input_resize, max_detected_persons, min_person_width, device) 

        self.skeleton_engine = SkeletonEngine(kp_model, kp_cfg, device)

        self.feature_engine = FeatureEngine(pr_model, self.LMK_VISIBILITY_THRESHOLD, device)

        self.BodyTracker = FishEyeTracker(pr_threshold, body_db_file)

        # self.queue_det = Queue(1)
        # self.queue_skl = Queue(1)
        # self.queue_feat = Queue(1)
        # self.rapid_det = Thread(target=self.RapidPersonDetect.worker, args=(self.queue_det))
        # self.rapid_det.daemon = True
        # self.rapid_det.start()


    def RapidPersonBoxTrackMatchFeature(self, bgrimg, dts, lmks, lmk_confs, body_crops, body_features):
        img_dim = bgrimg.shape[:2]
        # frame = rescale_frame(frame, percent=100)
        # rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
        # dts = self.detection_engine.RapidPersonDetect(rgbimg)

        self.frame_num += 1

        # if len(dts) == 0:
        #     return None    

        #fisheye camera
        if self.fisheye>0: 
            print("**FISHEYE**")

            # #skeleton detection
            # bboxs, lmks, lmk_confs, rot = self.skeleton_engine.SkeletonDetect(bgrimg, dts)

            # #feature extraction
            # body_crops, body_features = self.feature_engine.BodyFeatureExtraction(rgbimg, dts, lmks, lmk_confs)
            bboxs = dts[:,:4]

            #update tracking table
            t1 = time.time()
            trackids, subids = self.BodyTracker.update(bboxs, lmks, lmk_confs, body_features, img_dim)
            t2 = time.time() - t1
            print("tracking time ", t2)

        # else: #dome camera
        #     print("**DOME**")
        #     #detect body landmarks
        #     lmks, lmk_confs = self.kp.detect_batch(bgrimg, bboxs)
        #     #crop upper body
        #     body_crops, flags = self.__crop_rot_upper_body(rgbimg, dts)
        #     #extract body features
        #     body_features = self.__extract_body_features(body_crops)
        #     #crop faces
        #     face_crops = self.__align_and_crop(frame2, lmks)
        #     # #extract face features
        #     trackids, subids = self.BodyFaceTracker.update(bboxs, lmks, lmk_confs, body_features, face_crops, angles)

        visualization.draw_dt_on_np(bgrimg, dts)
        visualization.draw_lmks(bgrimg, lmks, lmk_confs, (255,255,255), self.LMK_VISIBILITY_THRESHOLD)

        return dts, body_crops, subids, trackids, lmks, bgrimg

    def RapidPersonBoxTrackMatchFeature_no_thread(self, bgrimg):
        img_dim = bgrimg.shape[:2]
        # frame = rescale_frame(frame, percent=100)
        rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
        dts = self.detection_engine.RapidPersonDetect(rgbimg)

        self.frame_num += 1

        if len(dts) == 0:
            return None    

        #fisheye camera
        if self.fisheye>0: 
            print("**FISHEYE**")

            #skeleton detection
            lmks, lmk_confs = self.skeleton_engine.SkeletonDetect(bgrimg, dts)

            #feature extraction
            body_crops, body_features = self.feature_engine.BodyFeatureExtraction(bgrimg, dts, lmks, lmk_confs)
            bboxs = dts[:,:4]

            #update tracking table
            t1 = time.time()
            trackids, subids = self.BodyTracker.update(bboxs, lmks, lmk_confs, body_features, img_dim)
            t2 = time.time() - t1
            print("tracking time ", t2)

        # else: #dome camera
        #     print("**DOME**")
        #     #detect body landmarks
        #     lmks, lmk_confs = self.kp.detect_batch(bgrimg, bboxs)
        #     #crop upper body
        #     body_crops, flags = self.__crop_rot_upper_body(rgbimg, dts)
        #     #extract body features
        #     body_features = self.__extract_body_features(body_crops)
        #     #crop faces
        #     face_crops = self.__align_and_crop(frame2, lmks)
        #     # #extract face features
        #     trackids, subids = self.BodyFaceTracker.update(bboxs, lmks, lmk_confs, body_features, face_crops, angles)

        visualization.draw_dt_on_np(bgrimg, dts)
        visualization.draw_lmks(bgrimg, lmks, lmk_confs, (255,255,255), self.LMK_VISIBILITY_THRESHOLD)

        return dts, body_crops, subids, trackids, lmks, bgrimg

    def worker(self, input_q, output_q):
        # detect_q = Queue()
        # skeleton_q = Queue()
        # feature_q = Queue()
        # detect_q.put(self.detection_engine.worker,

        while True:
            output = input_q.get()
            # bgrimg, dts = output
            bgrimg, dts, lmks, lmk_confs, body_crops, body_features = output
            # output_q.put()
            output_q.put(self.RapidPersonBoxTrackMatchFeature(bgrimg, dts, lmks, lmk_confs, body_crops, body_features))

    def SaveBodyDB(self):
        self.BodyFaceTracker.save_database(self.body_db_file)

class DetectionEngine(object):
    def __init__(self, pd_model, pd_det_threshold=0.5, pd_nms_threshold=0.3, pd_input_resize=0, 
        max_detected_persons=0, min_person_width=100, device=-1):

        self.min_person_width = min_person_width
        self.pd_input_resize = pd_input_resize

        #need to post process detection data
        self.conf_thresh = pd_det_threshold
        self.nms_thresh = pd_nms_threshold

        self.pd_model = pd_model

        if device<0:
            print("TENSORRT device negative")
            exit()

        #tensorrt engine initialization
        self.cfx = pcuda.Device(device).make_context()

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        print("MODEL ", self.pd_model, self.pd_input_resize)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.pd_model, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

        self.context = self.engine.create_execution_context()
        self.in_cpu, self.out_cpu, self.in_gpu, self.out_gpu, self.stream = self._alloc_buf(self.engine)

    #     self.__detector_rapid = RapidDetector(model_name='rapid',
    #         weights_path=pd_model, device=device, 
    #         pd_input_resize=pd_input_resize, max_persons=max_detected_persons,
    #         conf_thresh=pd_det_threshold, nms_thresh=pd_nms_threshold)

    # def RapidPersonDetect(self, bgrimg):
    #     rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
    #     pil_img = Image.fromarray(rgbimg)

    #     #output: in darknet format (cx,cy,w,h)
    #     t1 = time.time()
    #     detections = self.__detector_rapid.predict_pil(pil_img)

    #     #filter out small detections
    #     dts = self.__filterDts(detections)

    #     t2 = time.time() - t1
    #     print("Rapid Detection time ", t2)

    #     return dts

    def _get_img_tensor(self, pilimg):
        input_size = self.pd_input_resize

        img, _, pad_info = utils.rect_to_square(pilimg, None, input_size, 0)

        # img_tensor = transform(img)
        img_tensor = tvf.to_tensor(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)

        return img_tensor, pad_info

    def _alloc_buf(self, engine):
        # host cpu mem
        # h_in_size = trt.volume(self.engine.get_binding_shape(0))
        # h_out_size = trt.volume(self.engine.get_binding_shape(1))
        # h_in_dtype = trt.nptype(self.engine.get_binding_dtype(0))
        # h_out_dtype = trt.nptype(self.engine.get_binding_dtype(1))

        h_in_size = trt.volume(engine.get_binding_shape(0))
        h_out_size = trt.volume(engine.get_binding_shape(1))
        h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
        h_out_dtype = trt.nptype(engine.get_binding_dtype(1))

        in_cpu = pcuda.pagelocked_empty(h_in_size, h_in_dtype)
        out_cpu = pcuda.pagelocked_empty(h_out_size, h_out_dtype)

        # allocate gpu mem
        in_gpu = pcuda.mem_alloc(in_cpu.nbytes)
        out_gpu = pcuda.mem_alloc(out_cpu.nbytes)
        stream = pcuda.Stream()
        # print("engine size ", h_in_size, h_out_size)

        return in_cpu, out_cpu, in_gpu, out_gpu, stream

    def _inference(self, pilimg):
        img_tensor, pad_info = self._get_img_tensor(pilimg)
        print(pilimg.size)
        print(img_tensor.shape)
        img_array = img_tensor.numpy()

        self.cfx.push()

        # restore
        # stream  = self.stream
        context = self.context

        in_cpu = self.in_cpu
        in_gpu = self.in_gpu
        out_cpu = self.out_cpu
        out_gpu = self.out_gpu
        # bindings = self.bindings

        # context = engine.create_execution_context()
        # in_cpu, out_cpu, in_gpu, out_gpu, stream = self._alloc_buf(engine)

        t1 = time.time()
        # # async version
        # pcuda.memcpy_htod_async(in_gpu, img_array, stream)
        # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
        # pcuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
        # stream.synchronize()

        # sync version
        pcuda.memcpy_htod(in_gpu, img_array)
        context.execute(1, [int(in_gpu), int(out_gpu)])
        # context.execute(1, [int(in_gpu), int(out_gpu)])
        pcuda.memcpy_dtoh(out_cpu, out_gpu)

        self.cfx.pop()
        t2 = time.time() - t1
        print("TRT inference time ", t2)

        print(out_cpu)
        dts = torch.Tensor(self.out_cpu).reshape(-1,6)
        print("DTS ", dts[0])

        # post-processing, filter out detections
        dts = dts[dts[:,5] >= self.conf_thresh]
        print("DTS ", dts.shape)
        if len(dts) > 1000:
            _, idx = torch.topk(dts[:,5], k=1000)
            dts = dts[idx, :]

        dts = utils.nms(dts, is_degree=True, nms_thres=self.nms_thresh, img_size=self.pd_input_resize)
        dts = utils.detection2original(dts, pad_info.squeeze())

        return dts

    def RapidPersonDetect(self, bgrimg):
        t1 = time.time()

        # img_path = "/media/ubuntu/4A0076EA38BD8ECE/JK/KAJIMA/OrgRapid/RAPiD-master/images/exhibition.jpg"
        # frame = cv2.imread(img_path)
        # rgbimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

        pilimg = Image.fromarray(rgbimg)

        #output: in darknet format (cx,cy,w,h)
        # result = self._inference(img_array)
        # result = self._inference(img_array, context)
        dts = self._inference(pilimg)

        #filter out small detections
        detections = self.__filterDts(dts)

        t2 = time.time() - t1
        print("Rapid Detection time ", t2)

        return detections

    def __filterDts(self, dts):
        detections = dts
        #filter out detection which have width above threshold
        detections = detections[detections[:,2] >= self.min_person_width]

        # for i in range(len(dts)):
        #     bbox = dts[i]
        #     x,y,w,h = bbox[:4]
        #     if w<self.min_person_width:
        #         detections = detections[detections!=dts[i]]

        return detections

    def __filterDtsBatch(self, dts):
        filter_detection = []
        for i in range(len(dts)):
            if dts[i] is None:
                continue;
            detections = self.__filterDts(dts[i])
            filter_detection.append(detections)

        return filter_detection

    def worker(self, input_q, output_q):
        while True:
            bgrimg = input_q.get()
            dts = self.RapidPersonDetect(bgrimg)
            if len(dts) == 0:
                continue
            output_q.put((bgrimg, dts))
            # queue.task_done() # this is new 

class SkeletonEngine(object):
    def __init__(self, kp_model, kp_cfg, device=-1):

        # print("initializing lmk model", kp_model, kp_cfg)
        self.kp = HumanPoseDetection(kp_model, kp_cfg, device)

    def __get_angle_batch(self, imgw, imgh, detections):
        dir = []
        for bb in detections:
            x,y,w,h,angle = bb[:5]
            theta = self.__get_angle(imgw, imgh, x, y)
            dir.append(theta)

        return dir

    def __get_angle(self, imgw, imgh, x, y):
        # https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html
        top_x, top_y = imgw/2, 0
        center_x, center_y = imgw/2, imgh/2

        a = math.sqrt((top_x-x)**2 + (top_y-y)**2)
        b = math.sqrt((top_x-center_x)**2 + (top_y-center_y)**2)
        c = math.sqrt((center_x-x)**2 + (center_y-y)**2)

        A = math.acos((b**2 + c**2 - a**2) / (2*b*c))
        if x > center_x:
            return (math.pi + math.pi - A)

        return A 

    def SkeletonDetect(self, bgrimg, dts):
        t1 = time.time()
        imgh, imgw = bgrimg.shape[:2]
        bboxs = dts[:,:4]
        # angles = dts[:,4:5]
        # print("ANGLES", angles)            

        #get box angles
        rot = self.__get_angle_batch(imgw, imgh, dts)
        #detect body landmarks
        lmks, lmk_confs = self.kp.detect_batch(bgrimg, bboxs, rot)
        #crop upper body
        t2 = time.time() - t1
        print("Skeleton Detection time ", t2)

        return lmks, lmk_confs

    def worker(self, input_q, output_q):
        while True:
            bgrimg, dts = input_q.get()
            lmks, lmk_confs = self.SkeletonDetect(bgrimg, dts)
            output_q.put((bgrimg, dts, lmks, lmk_confs))

class FeatureEngine(object):
    def __init__(self, pr_model, LMK_VISIBILITY_THRESHOLD, device=-1):

        # Initializing person matching module
        self.body_emb_layer = "fc1_output"
        self.pr_size = 128
        self.LMK_VISIBILITY_THRESHOLD = LMK_VISIBILITY_THRESHOLD

        self.__ubody_embedding = UbodyEmbedding(model_path=pr_model,
                                             model_epoch=int(0),
                                             device=device,
                                             input_size=self.pr_size,
                                             emb_layer_name=self.body_emb_layer)

        # self.__fullbody_embedding = FullbodyEmbedding(model_name='osnet_ain_x1_0',
        #                                     model_path='./models/Body/osnet_ain_ms_d_c.pth.tar', 
        #                                     device=device)

        # mean body
        # self.mean_body = np.array([
        #         [28.0, 55.0],
        #         [100.0, 55.0],
        #         [28.0, 127.0],
        #         [100.0, 127.0]], dtype=np.float32)

        # self.mean_body = np.array([
        #         [32.0, 58.0],
        #         [96.0, 58.0],
        #         [32.0, 122.0],
        #         [96.0, 122.0]], dtype=np.float32)

        # #org mean body
        # self.mean_body = np.array([
        #         [64.0, 54.0],
        #         [64.0, 126.0]], dtype=np.float32)

        self.mean_body = np.array([
                [64.0, 50.0],
                [64.0, 127.0]], dtype=np.float32)

    def __crop_rot_upper_body(self, img, detections):

        imgh, imgw = img.shape[:2]

        chips = []
        flags = []
        for bb in detections:
            x,y,w,h,angle = bb[:5]
            print(x,y,w,h,imgh,imgw)
            if self.fisheye:
                theta = -self.__get_angle(imgw, imgh, x, y)

                c, s = np.cos(theta), np.sin(theta)

                R = np.asarray([[c, s], [-s, c]])
                pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
                rot_pts = []
                for pt in pts:
                    rot_pts.append(([x, y] + pt @ R).astype(int))

                contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])

                width = int(w)
                height = int(h)
                src_pts = contours.astype("float32")
                dst_pts = np.array([[0, 0],[width-1, 0], [width-1, height-1],
                            [0, height-1]], dtype="float32")

                tform = transform.SimilarityTransform()
                tform.estimate(src_pts, dst_pts)
                M = tform.params[0:2, :]
                chip = cv2.warpAffine(img, M, (self.pr_size, self.pr_size))
                flags.append(True)
            else:
                x1, y1 = int(x-w/2), int(y-h/2)
                x2, y2 = int(x1+w), int(y1+h)
                if x1<0:
                    x1=0
                if y1<0:
                    y1=0
                if x2>imgw:
                    x2 = imgw
                if y2>imgh:
                    y2 = imgh

                ub_width = x2-x1
                chip = img[y1:y1+ub_width,x1:x2].copy()
                chip = cv2.resize(chip, (self.pr_size, self.pr_size))
                flags.append(False)

            chips.append(chip)

        return chips, flags

    def __crop_upper_body(self, img, bbox):
        chips = []
        (H, W) = img.shape[:2]
        for bb in bbox:
            # print("BOUNDING BOX",bb)
            (x1, y1) = (bb[0], bb[1])
            (x2, y2) = (bb[2], bb[2])
            if x1<0:
                x1=0
            if y1<0:
                y1=0
            if x2>W:
                x2 = W
            if y2>H:
                y2 = H

            ub_width = x2-x1
            chip = img[y1:y1+ub_width,x1:x2].copy()
            # outfile = "test_ub_{}.jpg".format(self.count)
            # cv2.imwrite(outfile, chip)
            chip = cv2.resize(chip, (self.pr_size, self.pr_size))
            chip = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
            chips.append(chip)

        return np.array(chips)

    def __crop_rot_upper_body_lmk(self, input_img, detections, landmarks, lmk_confs):

        img = input_img.copy()
        imgh, imgw = input_img.shape[:2]

        chips = []
        flags = []
        for i in range(len(detections)):
            lmk = landmarks[i]
            confs = lmk_confs[i]

            left_hip, left_hip_conf = lmk[11], confs[11]
            right_hip, right_hip_conf = lmk[12], confs[12]
            middle_of_hip = [(left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2]
            middle_of_hip_conf = 0 if left_hip_conf < self.LMK_VISIBILITY_THRESHOLD or right_hip_conf < self.LMK_VISIBILITY_THRESHOLD else 1

            left_shoulder, left_shoulder_conf = lmk[5], confs[5]
            right_shoulder, right_shoulder_conf = lmk[6], confs[6]
            middle_of_shoulder = [(left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2]
            middle_of_shoulder_conf = 0 if left_shoulder_conf < self.LMK_VISIBILITY_THRESHOLD or right_shoulder_conf < self.LMK_VISIBILITY_THRESHOLD else 1

            if middle_of_shoulder_conf>self.LMK_VISIBILITY_THRESHOLD  and middle_of_hip_conf>self.LMK_VISIBILITY_THRESHOLD:
                print(colored("CROPPING WITH LMK", "blue"))

                src_pts = np.array([[middle_of_shoulder[0],middle_of_shoulder[1]],
                       [middle_of_hip[0],middle_of_hip[1]]], dtype=np.float32)

                tform = transform.SimilarityTransform()
                tform.estimate(src_pts, self.mean_body)
                M = tform.params[0:2, :]
                warped = cv2.warpAffine(img, M, (self.pr_size, self.pr_size), borderValue=0.0)
                chips.append(warped)
                flags.append(True)
            else:
                bb = detections[i]
                x,y,w,h,angle = bb[:5]
                print(colored("CROPPING WITH AFFINE", "red"))

                theta = -rot[i]

                c, s = np.cos(theta), np.sin(theta)

                R = np.asarray([[c, s], [-s, c]])
                pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, -h/2+w/2], [-w/2, -h/2+w/2]])
                rot_pts = []
                for pt in pts:
                    rot_pts.append(([x, y] + pt @ R).astype(int))

                contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])

                width = int(w)
                height = int(h)
                src_pts = contours.astype("float32")
                dst_pts = np.array([[0, 0],[self.pr_size-1, 0], [self.pr_size-1, self.pr_size-1],
                            [0, self.pr_size-1]], dtype="float32")

                tform = transform.SimilarityTransform()
                tform.estimate(src_pts, dst_pts)
                M = tform.params[0:2, :]
                chip = cv2.warpAffine(img, M, (self.pr_size, self.pr_size))
                chips.append(chip)
                flags.append(True)

        return chips, flags

    def BodyFeatureExtraction(self, bgrimg, dts, lmks, lmk_confs):
        t1 = time.time()
        rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
        body_crops, flags = self.__crop_rot_upper_body_lmk(rgbimg, dts, lmks, lmk_confs)
        #extract body features
        body_features = self.__ubody_embedding.extract_feature_batch(body_crops)
        # body_features = self.__fullbody_embedding(body_crops)
        t2 = time.time() - t1
        print("feature ectraction time ", t2)

        return body_crops, np.array(body_features)

    def worker(self, input_q, output_q):
        while True:
            bgrimg, dts, lmks, lmk_confs = input_q.get()
            body_crops, body_features = self.BodyFeatureExtraction(bgrimg, dts, lmks, lmk_confs)
            output_q.put((bgrimg, dts, lmks, lmk_confs, body_crops, body_features))


    
