import sys
import os
import json
import logging
import pathlib
import threading
import datetime as dt

from camera2 import CamHolder, FloorHolder, MicHolder, DetectionEngine, CameraStream

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))
from adaptor import Adaptor
from jsonutils import json2str
from localization import Localize

# camera class inherit with Adaptor for Redis communication
class Camera(Adaptor):
    def __init__(self, args, **kw):
        self.args = args
        self.cfg = {}
        self.subscribe_channels = [
            'camera.{}.update'.format(self.args.id),
            'camera.{}.sqlquery'.format(self.args.id)
            ]
        self.pcid = self.args.pcid
        self.basefile = None
        Adaptor.__init__(self, args, **kw)
        self.load_config()
        self.start_listen_bus()
        self.run()

    # load config msg into local buffer
    def load_config (self):
        logging.debug('Load configuration from Redis for {}'.format(self.args.id))
        cfg = Adaptor.get_redismsg_by_channel(self, '{}.config'.format(self.component_prefix))
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
    
    # redis message listener
    def process_redis_msg (self, ch, msg):
        if ch in self.subscribe_channels:
            if 'sqlquery' in ch:
                logging.debug('[{}]: channel: {}'.format(self.args.id, ch))
            else:
                logging.debug('[{}]: channel: {} ==> {}'.format(self.args.id, ch, msg))
            if 'update' in ch:
                # if msg: {'config': True}, update cfg
                if msg.get('config', False):
                    self.load_config()
            if 'sqlquery' in ch:
                self.process_sql_result(msg)
                self.run_init()

    # container for kajima camera
    def run (self):
        # getting camera configuration
        msg = Adaptor.get_redismsg_by_channel(self, '{}.detail-config'.format(self.component_prefix))
        if msg is None:
            logging.debug('Detail config not found, make request for detail config to backend server')
            self.publish_redis_msg('{}.query'.format(self.component_prefix), {'msgType': 'init'})
        else:
            logging.debug('Found detail config in redis bus')
            self.process_sql_result(msg)
            self.run_init()
        
    def run_init (self):
        # intialization
        self.process_localization()
        self.process_stream()
        self.process_engine()

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
    def process_engine (self):
        self.cur_engine = DetectionEngine(self.cfg)
        logging.debug('Detection Engine module initialized...')

    # initialize camera stream
    def process_stream (self):
        self.stream = CameraStream(
            cam_width=self.cfg.get('camera_width', None),
            cam_height=self.cfg.get('camera_height', None),
            cam_fps=self.cfg.get('camera_fps', None),
            src=self.localize.camera.source
        )
        self.stream.start()
        logging.debug('Camera Stream module initialized...')
    
    # thread loop for process image and result publish
    def cam_run (self):
        import time
        while True:
            try:
                
                _success, _img = self.stream.read()
                # _success = False
                if not _success:
                    logging.error('Unable to retrieve Camera Stream image')
                    continue
            
                
                _output = self.cur_engine.run(_img)
                
                # _output = False
                # logging.info("###{}###{}######".format(_output[-2],len(_output[-2])))
                if _output:
                    
                    logging.debug('Reading Image Start Timing...')
                    res = {}
                    start_time = time.time()
                    boxes, person_crop, face_labels, tids, lmks, image, pmvs, actions  = _output
                    # logging.debug("###########{}##########".format(len(tids)))
                    res, t = self.localize.find_location(_output)
                    

                    # res format should ne {'dt': [], 'dt': []}
                    # convert into [{'list': [], 'timestamp': ''}, {'list': [], 'timestamp': ''}]
                    resList = []
                    #print (res)
                    for k, v in res.items():
                        list = [sublist + [str(num)] for sublist, num in zip(v, _output[-2])]
                        resList.append({'list': list, 'timestamp': t})

                    logging.debug(resList[-1])

                    _res = {'timestamp': dt.datetime.now(), 'result': resList, 'pcid': self.pcid}
                    # logging.debug("Continue to process but no publish")
                    # continue
                    self.redis_conn.publish(
                        '{}.result'.format(self.component_prefix),
                        json2str(_res)
                    )

                    msg = {
                        "cam_id": self.args.id,
                        "timestamp": {"$dt": dt.datetime.now().timestamp()},
                        "pcid": 7000,
                        "people_count": len(tids) if _output is not None else 0
                    }          
                    self.redis_conn.publish("util.{}.query".format(self.args.id),json2str(msg))
                    # time.sleep(60)
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
    add_common_adaptor_args(
        parser, 
        type='fish-eye', 
        location='Office-1',
        id='camera-18',
        pcid=7000)
    args = au.parse_args(parser)

    cam = Camera(args=args)
    logging.debug("{:all}".format(cam))
    try:
        while not cam.is_quit(1):
            pass
    except KeyboardInterrupt:
        cam.close()

