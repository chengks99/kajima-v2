#!/bin/bash
#CUDA_VISIBLE_DEVICES=1 python3 kajima_camera2_ROI_offline.py --redis-host 192.168.0.251 --redis-passwd ew@icSG23 --id 50 --pcid 7000 -d -v
CUDA_VISIBLE_DEVICES=1 python3 kajima_camera2_offline_dome.py --redis-host 192.168.0.251 --redis-passwd ew@icSG23 --id 50 --pcid 7000 -d -v
