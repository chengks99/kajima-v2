import sys
import os
import json
import logging
import pathlib
import threading
import datetime as dt
import cv2
from camera2 import CamHolder, FloorHolder, MicHolder, DetectionEngine, CameraStream
from shapely.geometry import Polygon,Point
from shapely.ops import nearest_points
scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))
from adaptor import Adaptor
from jsonutils import json2str
from localization import Localize
import numpy as np
import copy
#-------------------- Extract the majority number from utility rate --------------

def majority_element(nums):
    count_ = 0
    candidate = None

    for num in nums:
        if count_ == 0:
            candidate = num
            count_ = 1
        elif num == candidate:
            count_ += 1
        else:
            count_ -= 1

    return candidate

class BGRColor:
    WHITE   = (255, 255, 255)
    SILVER  = (192, 192, 192)
    GRAY    = (128, 128, 128)
    GREY    = (128, 128, 128)
    BLACK   = (0, 0, 0)

    BLUE    = (255, 0, 0)
    NAVY    = (128, 0, 0)
    LIME    = (0, 255, 0)
    GREEN   = (0, 128, 0)
    RED     = (0, 0, 255)
    MAROON  = (0, 0, 128)

    YELLOW  = (0, 255, 255)
    OLIVE   = (0, 128, 128)
    GOLD    = (0, 215, 255)

    CYAN    = (255, 255, 0)
    AQUA    = (255, 255, 0)
    TEAL    = (128, 128, 0)

    MAGENTA = (255, 0, 255)
    FUCHSIA = (255, 0, 255)
    PURPLE  = (128, 0, 128)

    CUSTOM_COLOR = (35, 142, 107)

def remove_outside_bbox(ROI,dets) :    

    cx = int(dets[0])
    cy = int(dets[1])
    hw = int(dets[2]/2)
    hh = int(dets[3]/2)

    x0 = cx - hw
    y0 = cy - hh
    x1 = cx + hw
    y1 = cy + hh

    
    dets = [[int(x0),int(y0)],[int(x0),int(y1)],[int(x1),int(y1)],[int(x1),int(y0)]]
    ROI = ROI.tolist()
    dets = [tuple(i) for i in dets]
    ROI = [tuple(i) for i in ROI]
    detection_polygon = Polygon(dets)
    ROI_polygon = Polygon(ROI)
    deleted = -1
    intersect = detection_polygon.intersection(ROI_polygon).area
    detection_area = detection_polygon.area
    iou = intersect / detection_area
    if iou >= 0.1 :
        deleted = 0
        return deleted
    return deleted

def converted_loc(ROI,loc) :    


    ROI = ROI.tolist()
    ROI_polygon = Polygon(ROI)
    new_loc_res = {}
    for k,v in loc.items() :
        new_loc_res[k] = []

    for k,v in loc.items() :
        for point in v :
            cur_loc_point = Point(point[0],point[1])
            distance = ROI_polygon.distance(cur_loc_point) 
            if distance == 0 :
                ## Inside ROI 
                new_loc_res[k].append([point[0],point[1],point[2]])
            else :

                ## Outside ROI 
                p1, p2 = nearest_points(ROI_polygon, cur_loc_point)

                ## The point is p1 , converted to string, convert to float
                p1_x = float(p1.x)
                p1_y = float(p1.y)

                new_loc_res[k].append([p1_x,p1_y,point[2]])

    return new_loc_res

def draw_box(image, box, label='', color=BGRColor.CYAN, font_size=0.7):
    """
        image: image matrix (h, w, c)
        box: numpy array of shape (n x 5). [x1, x2, y1, y2, score]
        label: text to put above box
        color: BGR colour value for box and text label colour
        font_size: Size of text label
    """
    img = image.copy()
    #Object det box: cx, cy, w, h, angle, conf

    # box[0] = box[0] - box[2]/2
    # box[1] = box[1] - box[3]/2

    bbox = list(map(int,box))
    # Draw bounding box and label:
    # label = '{}_{}'.format(label, bbox[2])
    textsize = cv2.getTextSize(label, 0, font_size, thickness=3)[0]
    # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color=color, thickness=2)
    cv2.rectangle(img, (bbox[0], bbox[1]-5-textsize[1]), (bbox[0]+textsize[0], bbox[1]-5), color=BGRColor.PURPLE, thickness=-1)
    cv2.putText(img, label, (bbox[0], bbox[1] - 5), fontFace=0, fontScale=font_size, color=BGRColor.YELLOW, thickness=4)
    cv2.putText(img, label, (bbox[0], bbox[1] - 5), fontFace=0, fontScale=font_size, color=color, thickness=3)

    # print(bbox[2]-bbox[0])
    return img

def overlay_text(img, font_size, *texts):
    # img = image.copy()
    for i, text in enumerate(texts):
        cv2.putText(img, text, (10, (i+1)*50), 0, font_size, color=BGRColor.RED, thickness=3)
        cv2.putText(img, text, (10, (i+1)*50), 0, font_size, color=BGRColor.WHITE, thickness=2)
    return img

# camera class inherit with Adaptor for Redis communication
class Camera(Adaptor):
    def __init__(self, args, **kw):
        self.args = args
        self.cfg = {}
        self.subscribe_channels = [
            'camera.{}.update'.format(self.args.id),
            'camera.{}.sqlquery'.format(self.args.id),
            'person.body.updates',
            'http.data.int'
            ]
        self.pcid = self.args.pcid
        self.utility = self.args.utility
        self.view = self.args.view
        self.basefile = None
        Adaptor.__init__(self, args, **kw)
        # self.devid = self.args.devid
        self.load_config()
        self.start_listen_bus()
        self.run()
        self.ROI = np.array(self.cfg.get('cam_roi', [0,0,0,2991,2991,2991,2991,0]), dtype=np.int)
        ROI_length = len(self.ROI) // 2
        self.ROI = self.ROI.reshape(ROI_length,-1)

        self.ROI_loc = np.array(self.cfg.get('loc_roi', [795,704,795,3013,5800,3013,5800,704]), dtype=np.int)
        ROI_length = len(self.ROI_loc) // 2
        self.ROI_loc = self.ROI_loc.reshape(ROI_length,-1)
            
    # load config msg into local buffer
    def load_config (self):
        logging.debug('Load configuration from Redis for {}'.format(self.args.id))
        cfg = Adaptor.get_redismsg_by_channel(self, '{}.config'.format(self.component_prefix))
        # cfg = None
        if cfg is None:
            logging.debug('No Configuration for {} found in Redis, use config file {} instead'.format(self.args.id, self.args.config))
            if not os.path.isfile(self.args.config):
                logging.error('Unable to retrieve configuration file ...')
                exit(1)
            with open (self.args.config) as cfgf:
                self.cfg = json.load(cfgf)
        else:
            logging.debug('Configuration in redis loaded ...')
            if 'pcid' in cfg: self.pcid = cfg['pcid']
            if 'devid' in cfg: self.devid = cfg['devid']
            if 'config' in cfg:
                self.cfg = cfg['config']
            else:
                for key, val in cfg.items():
                    if key == 'id': continue
                    cfgPath = str(scriptpath.parent / val)
                    if not os.path.isfile(cfgPath):
                        logging.error('Unable to retrieve {} config from path: {}'.format(key, val))
                        exit(1)
                    with open(val) as cfgf:
                        self.cfg.update(json.load(cfgf))

            
        logging.debug('{} configuration: {}'.format(self.args.id, self.cfg))
        # exit(1)
    # redis message listener
    def process_redis_msg (self, ch, msg):
        if ch in self.subscribe_channels:
            # if 'sqlquery' in ch:
            #     logging.debug('[{}]: channel: {}'.format(self.args.id, ch))
            # else:
            #     logging.debug('[{}]: channel: {} ==> {}'.format(self.args.id, ch, msg))
            if 'update' in ch and not 'person' in ch:
                # if msg: {'config': True}, update cfg
                if msg.get('config', False):
                    self.load_config()
            if 'sqlquery' in ch:
                self.process_sql_result(msg)
                self.run_init()
            if ch == 'person.body.updates':
                # print('[{}]: channel: {}'.format(self.args.id, ch))
                print("Received Body Updates")
                try:
                    self.cur_engine.person_body_updates(msg)
                except:
                    pass
            if ch == 'http.data.int':
                self.process_env_data(msg)

    # container for kajima camera
    def run (self):
        # getting camera configuration
        msg = Adaptor.get_redismsg_by_channel(self, '{}.detail-config'.format(self.component_prefix))
        # msg = None
        if msg is None:
            logging.debug('Detail config not found, make request for detail config to backend server')
            self.publish_redis_msg('{}.query'.format(self.component_prefix), {'msgType': 'init'})
        else:
            logging.debug('Found detail config in redis bus')
            self.process_sql_result(msg)
            self.run_init()
        
    def run_init (self):
        body_details = Adaptor.get_redismsg_by_channel(self,'person.body.features')
        if body_details is None:
            print('Empty body features')
            # body_details = {'fvList': []}
            body_details = None
        self.process_engine(body_details)
        self.process_localization()
        self.process_stream()

        # start thread
        self.th_quit = threading.Event()
        self.th = threading.Thread(target=self.cam_run)
        self.th.start()
    
    # process sql messaging
    def process_sql_result (self, msg):
        if msg.get('type', None) == 'init':
            self.cam = CamHolder(self.args.id, msg.get('cam', None))
            self.floor = FloorHolder(self.args.id, msg.get('floor', None))
            self.mic = MicHolder(self.args.id, msg.get('mic', None))

    
    #!FIXME: need to import correct localization file
    def process_localization (self):
        self.localize = Localize(self.cam)
        logging.debug('Localization module initialized...')

    # initialize detection engine
    def process_engine (self,body_details):
        self.cur_engine = DetectionEngine(self.cfg,body_details)
        logging.debug('Detection Engine module initialized...')

    # initialize camera stream
    def process_stream (self):

        ## Old Code commented out to test JK's Code -- 20231026
        self.stream = CameraStream(
            cam_width=self.cfg.get('camera_width', None),
            cam_height=self.cfg.get('camera_height', None),
            cam_fps=self.cfg.get('camera_fps', None),
            src=self.localize.camera.source
        )
        self.stream.start()

        # self.stream = cv2.VideoCapture(self.localize.camera.source)
        # self.stream.set(3,self.cfg.get('camera_width',640))
        # self.stream.set(4,self.cfg.get('camera_height',480))
        # self.stream.set(5,self.cfg.get('camera_fps',30))

        logging.debug('Camera Stream module initialized...')

    # process env data
    def process_env_data (self, msg):

        logging.info("MESSAGE RECEIVED THROUGHT REDIS CHANNEL")
        logging.info(msg)
        #print('**********************************{}'.format(self.devid))
        #print (msg)
        _dataMsg = msg.get('data', {})
        #print ('111111+++ {}'.format(_dataMsg))
        _data = _dataMsg.get(str(self.devid), {})
        #print ('************** DevID: {}, data: {}'.format(self.devid, _data))

        if not _data == {}:
            _temp = _data.get('T', {}).get('value', -1)
            _hum = _data.get('H', {}).get('value', -1)
            #print (_temp, _hum)
            if _temp > 0 and _hum > 0:
                if self.cur_engine.flag:
                    logging.debug('Env. data T: {}, H: {}'.format(_temp, _hum))
                    #print('Env. data T: {}, H: {}'.format(_temp, _hum))
                    self.cur_engine.set_env_var(_temp, _hum)

    # thread loop for process image and result publish
    def cam_run (self):
        if self.view:
            import cv2
        
        import time
        last_report = time.time()
        print("Test Run")
        self.utility_list = []
        while not self.is_quit(1):
            try:
                loop_start_time = time.time()
                _success, _img = self.stream.read()
                # _success = False
                if not _success:
                    logging.error('Unable to retrieve Camera Stream image')
                    continue
            
                
                _output = self.cur_engine.run(_img)
                # _output = None
                # _output = False
                # logging.info("###{}###{}######".format(_output[-2],len(_output[-2])))
                start_time = time.time()
                if _output:
                    logging.debug('Reading Image Start Timing...')
                    res = {}

                    boxes, person_crop, face_labels, tids, lmks, image, pmvs, actions, score, clothids   = _output
                    if boxes is None:
                        image = _img
                        tids = []
                        resList = [{"list" : []}]
                        logging.debug("NO DETECTION")
                        logging.debug("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
                    else:

                        # logging.debug("###########{}##########".format(len(tids)))
                        # old_boxes = copy.deepcopy(boxes)
                        index2delete = []
                        # logging.debug("ALL BOXES ARE : {}".format(str(boxes)))
                        for i in range(len(boxes)) :
                            removed = remove_outside_bbox(self.ROI,boxes[i])
                            # logging.debug("THE DECISION IS :{}".format(removed))
                            if removed == -1 :
                                index2delete.append(i)
                            else :
                                continue
                        index2delete = set(index2delete)
                        boxes = [v for i, v in enumerate(boxes) if i not in index2delete]
                        face_labels = [v for i, v in enumerate(face_labels) if i not in index2delete]
                        tids = [v for i, v in enumerate(tids) if i not in index2delete]
                        pmvs = [v for i, v in enumerate(pmvs) if i not in index2delete]
                        if len(boxes) > 0 :
                            # logging.debug("NEW BOXES ARE : {}".format(str(boxes)))
                            # logging.debug("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
                            _output = boxes, person_crop, face_labels, tids, lmks, image, pmvs, actions

                            res, t, time_used = self.localize.find_location(_output)

                            ## Implementation of new ROI in here :
                            res = converted_loc(self.ROI_loc,res)
                            # res format should ne {'dt': [], 'dt': []}
                            # convert into [{'list': [], 'timestamp': ''}, {'list': [], 'timestamp': ''}]
                            resList = []
                            #print (res)
                            for k, v in res.items():
                                if k == time_used : 
                                    list = [sublist + [str(num)] for sublist, num in zip(v, _output[-2])]
                                    resList.append({'list': list, 'timestamp': t})
                            # logging.debug(res)
                            # logging.debug(len(_output[3]))
                            # logging.debug(_output[3])
                            # logging.debug(resList[0]['list'])
                            # logging.debug("Track ID {}".format(tids))
                            # logging.debug("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                            _res = {'timestamp': dt.datetime.now(), 'result': resList, 'pcid': self.pcid}
                            # logging.debug("Continue to process but no publish")
                            # continue
                            self.redis_conn.publish(
                                '{}.result'.format(self.component_prefix),
                                json2str(_res)
                            )
                        else :
                            resList = [{"list" : []}]
                            logging.debug("THERE ARE NO NEW BOXES")
                    
                else : 
                    image = _img
                    tids = []
                    resList = [{"list" : []}]
                    # resList[0]['list']
                    # logging.debug("ROI :{}".format(str(self.ROI)))
                    # logging.debug(self.ROI)
                    # logging.debug(type(self.ROI))
                    # logging.debug(self.ROI.shape)
                    logging.debug("NO DETECTION")
                    logging.debug("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
                if self.view:
                    cv2.namedWindow('Video feed', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Video feed", 1280, 720)
                    cv2.moveWindow('Video feed', 50, 50)
                    logging.debug("CREATE WINDOW")
                    count = 1
                    avg_frame_t = 0
                    avg_fps = 0
                    avg_time = 0

                    for i in range(len(tids)):
                        print("tids : {}, face: {}".format(tids[i], face_labels[i]))
                        if tids[i]==-1:
                            continue

                        # label = "tid_"+str(tids[i])+"_"+face_labels[i] 
                        # label = face_labels[i] 
                        label = "tid_"+str(tids[i])+"_"+face_labels[i]+"_pmv:{:.2f}".format(pmvs[i]) 
                        # if face_labels[i] == 'UnK':
                        if "Unk" in face_labels[i]:
                            box_color = BGRColor.RED
                        else:
                            box_color = BGRColor.WHITE
                        logging.debug("-----------------------------------------")
                        # logging.debug(boxes[i])
                        image = draw_box(image, boxes[i], label, color=box_color, font_size=2.0)
                        logging.debug("DRAW TIDS")
                    # logging.debug("SKIP TIDS")
                    loop_dur = time.time() - loop_start_time
                    calc_fps = 1 / loop_dur
                    info_text = "Run time: {:.3f} s".format(time.time() - start_time)
                    loop_t_text = "Frame time: {:.3f} s".format(loop_dur)
                    fps_text = "Calculated FPS: {:.3f}".format(calc_fps)
                    logging.debug("CALCULATION of FPS")
                    image = overlay_text(image, 1.0, info_text, loop_t_text, fps_text)
                    # logging.debug("ROI is {}".format(str(self.ROI)))
                    cv2.polylines(image, [self.ROI], True, color=BGRColor.CYAN, thickness=10)
                    logging.debug("OVERLAY TEXT OVER IMAGE")
                    # For statistics:
                    count += 1
                    avg_frame_t = avg_frame_t + ((loop_dur - avg_frame_t) / count)
                    avg_fps = avg_fps + ((calc_fps - avg_fps) / count)
                    # cv2.imwrite("sample.png",image)
                    logging.debug("DRAW IMAGE INTO WINDOW")
                    cv2.imshow("Video feed", image)
                    if self.view  and cv2.waitKey(1) & 0xFF == ord('q'):
                        # video_shower.stop()
                        # change waitKey() for different refresh interval; press 'q' key to quit
                        cv2.destroyAllWindows()
                        self.view = False
                # logging.info("CHECK RES LIST")
                # logging.info(resList)

                self.utility_list.append(len(resList[0]['list']) if resList is not None else 0)
                
                if self.utility and time.time() - last_report >= 60 :
                    logging.info(self.utility_list)
                    ppl_count = majority_element(self.utility_list)
                    logging.info(ppl_count)
                    if ppl_count == 0 :
                        last_report = time.time()
                        self.utility_list = []
                        continue

                    msg = {
                    "cam_id": self.args.id,
                    "timestamp": {"$dt": dt.datetime.now().timestamp()},
                    "pcid": 7000,
                    "people_count": ppl_count
                    }    
                    logging.debug("Update count for utility: {}".format(json2str(msg)))
                    self.redis_conn.publish("util.{}.query".format(self.args.id),json2str(msg))
                    last_report = time.time()
                    self.utility_list = [] 
                


                continue
                # logging.debug(_res)
                logging.debug('Pubish result with {} number of key'.format(len(_res['result'])))
                # logging.debug("###########{}##########".format(_res['result']))
                process_time = time.time() - start_time
                logging.debug("Time needed for 1 process is {}".format(str(process_time)))
            except:
                continue
            # else:
            #    logging.error('Current stream produce None outcome from detection engine')
            if self.th_quit.is_set():
                break
            '''
            if loc:
                loc['timestamp'] = dt.datetime.now()
                self.redis_conn.publish(
                    '{}.result'.format(self.component_prefix), 
                    json2str(loc)
                )
            if self.th_quit.is_set():
                break
            '''

    '''
    # process image operation for loc recognition
    def process_image (self):
        _success, _img = self.stream.read()
        if not _success:
            logging.error('Unable to retrieve Camera Stream image')
            return

        _output = self.cur_engine.run(_img)

        if _output:
            return self.localize.find_location(_output)
            boxes, person_crop, face_labels, tids, lmks, image = _output
            localize = []

            for i in range(len(tids)):
                label = face_labels[i]
                box_color = BGRColor.RED if face_labels[i] == 'UnK' else BGRColor.WHITE
                logging.debug('Got output 2')

                _img = draw_box(_img, boxes[i], label, color=box_color, font_size=2.)
                _loc = [i, tids[i], face_labels[i]]
                bb = boxes[i]
                lmk = lmks[i]

                for bbox in range(len(bb)): _loc.append(bb[bbox].items())
                for point in range(len(lmk)):
                    _loc.append(lmk[point, 0])
                    _loc.append(lmk[point, 1])
                localize.append(_loc)

            if len(localize) == 0:
                logging.debug('Empty localize array ...')
                return {}
            else:
                logging.debug('Found localization info')
                return self.localize.find_location(localize)
        else:
            logging.debug('Unable to retrieve output from detection engine')
            return {}
    '''

    # periodically update adaptor status to server
    def get_status (self):
        r = Adaptor.get_status(self)
        r['status'] = 'Normal'
        return r
    
    # getting adaptor information
    def get_info (self):
        r = Adaptor.get_info(self)
        return r
    
    # close loop
    def close (self):
        self.th_quit.set()

if __name__ == "__main__" :
    import argsutils as au
    from adaptor import add_common_adaptor_args

    parser = au.init_parser('Kajima Camera Application')
    au.add_arg(parser, '-u','--utility', h='Running for ultility rate  {D}', a = True)
    au.add_arg(parser, '-v','--view', h='Debug view mode  {D}', a = True)    
    add_common_adaptor_args(
        parser, 
        type='fish-eye', 
        location='Office-1',
        id= 18,
        pcid=7000)
    args = au.parse_args(parser)
    args.config = str(scriptpath /'config/camera-{}.json'.format(args.id))
    logging.debug(args)

    cam = Camera(args=args)
    logging.debug("{:all}".format(cam))
    try:
        while not cam.is_quit(1):
            pass
    except KeyboardInterrupt:
        cam.close()

