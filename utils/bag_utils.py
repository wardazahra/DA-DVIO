import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from io import StringIO
import sys
from PIL import Image
import io
import tqdm as tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from utils.event_utils import compute_ms_to_idx
import h5py


def read_first_evs_from_rosbag(bag, evtopic):
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            t0_us = ev.ts.to_nsec()/1e3
            break
        break
    return t0_us


def read_evs_from_rosbag(bag, evtopic, H=180, W=240):
    print(f"Start reading evs from {evtopic}")

    evs = []
    progress_bar = tqdm.tqdm(total=bag.get_message_count(evtopic))
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            p = 1 if ev.polarity else 0
            evs.append([ev.x, ev.y, ev.ts.to_nsec()/1e3, p])
            # assert ev.x < W and ev.y < H # DEBUG
        progress_bar.update(1)

        # if len(evs) > 1000:
        #     break
    return np.array(evs) # (N, 4)


def read_evs_from_rosbag_witht0(bag, evtopic, t0_us=0, H=180, W=240):
    print(f"Start reading evs from {evtopic}")

    evs = []
    progress_bar = tqdm.tqdm(total=bag.get_message_count(evtopic))
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            p = 1 if ev.polarity else 0
            evs.append([ev.x, ev.y, ev.ts.to_nsec()/1e3-t0_us, p])#减去t0_us
            # assert ev.x < W and ev.y < H # DEBUG
        progress_bar.update(1)

        # if len(evs) > 1000:
        #     break
    return np.array(evs) # (N, 4)


def read_and_saved_evs_from_rosbag(bag, evtopic, H=180, W=240, t0=0,h5outfile='evs.h5'):
    print(f"Start reading evs from {evtopic}")

    evs = []
    progress_bar = tqdm.tqdm(total=bag.get_message_count(evtopic))
    first_message = True  # 用来检测第一组数据的标志

    with h5py.File(h5outfile, 'w') as f:
        for topic, msg, t in bag.read_messages(evtopic):
            evs = np.array([[ev.x, ev.y, ev.ts.to_nsec()/1e3-t0, 1 if ev.polarity else 0] for ev in msg.events])

            if not (evs[:, 2] >= 0).all():
                progress_bar.update(1)
                continue
            
            # make sure event timestamps is in ascending order
            if np.diff(evs[:, 2]).min() < 0:
                sorted_ied = np.argsort(evs[:, 2])
                evs = evs[sorted_ied]

            # 保存evs到文件
            if first_message==True:
                # 初始化并创建
                event_grp = f.create_group('/events')
                event_grp.create_dataset('x', shape=(0,), maxshape=(None,), dtype='<u2')
                event_grp.create_dataset('y', shape=(0,), maxshape=(None,), dtype='<u2')
                event_grp.create_dataset('t', shape=(0,), maxshape=(None,), dtype='<u4')
                event_grp.create_dataset('p', shape=(0,), maxshape=(None,), dtype='|u1')
                f.create_dataset('ms_to_idx', shape=(0,), maxshape=(None,), dtype="<u8")
                last_ms_end = evs[:, 2][0] // 1e3
                index_bias = 0
            elif first_message==False:
                event_grp = f['/events']
            num_events = evs.shape[0]
            event_grp['x'].resize(event_grp['x'].shape[0] + num_events, axis=0)
            event_grp['y'].resize(event_grp['y'].shape[0] + num_events, axis=0)
            event_grp['t'].resize(event_grp['t'].shape[0] + num_events, axis=0)
            event_grp['p'].resize(event_grp['p'].shape[0] + num_events, axis=0)
            event_grp['x'][-num_events:] = evs[:, 0]
            event_grp['y'][-num_events:] = evs[:, 1]
            event_grp['t'][-num_events:] = evs[:, 2]
            event_grp['p'][-num_events:] = evs[:, 3]

            # ms_to_idx = last_index+compute_ms_to_idx((evs[:, 2]*1e3-last_t*1e3), ms_start=num_ms)#转回ns(以上一次截止的时间为基准)
            ms_to_idx = compute_ms_to_idx((evs[:, 2]*1e3), ms_start=last_ms_end)#转回ns(以上一次截止的时间为基准)
            last_ms_end = last_ms_end + ms_to_idx.shape[0] - 1  # Advoid lack of previous events
            if not first_message:
                ms_to_idx = ms_to_idx[1:]
            
            num_ms = ms_to_idx.shape[0]  # -1 #去掉第一个0
            f['ms_to_idx'].resize(f['ms_to_idx'].shape[0] + num_ms, axis=0)
            f["ms_to_idx"][-num_ms:] = ms_to_idx + index_bias  #将新的ms_to_idx添加到文件中

            # f["ms_to_idx"][-num_ms:] = ms_to_idx[1:]#将新的ms_to_idx添加到文件中
            # last_index=f["ms_to_idx"][-1]
            index_bias += evs[:, 2].shape[0]
            if first_message:
                first_message = False
            progress_bar.update(1)

    return np.array(evs) # (N, 4)


def read_H_W_from_bag(bag, imgtopic):
    for topic, msg, t in bag.read_messages(imgtopic):
        H, W = msg.height, msg.width
        print(f"Read H, W from bag: {H}, {W}")
        return H, W

def read_images_from_rosbag(bag, imgtopic, H=180, W=240):
    imgs = []
    
    progress_bar = tqdm.tqdm(total=bag.get_message_count(imgtopic))
    for topic, msg, t in bag.read_messages(imgtopic):
        img_str = str(msg)
        img_str = img_str[img_str.find("data")+6:]
        img_str = img_str[1:-1].split(',')
        pixel_values = [int(v) for v in img_str]
        image_array = np.array(pixel_values, dtype=np.uint8)
        image_array = image_array.reshape((msg.height, msg.width))
        imgs.append(image_array)
        progress_bar.update(1)

        if abs(H- msg.height) > 2 or abs(W-msg.width) > 2:
            print(f"WARNING: H, W mismatch: {msg.height}, {msg.width}, {H}, {W}")    

        # if len(imgs) > 50: # TODO: remove!
        #     break
    return imgs

def read_rgb_images_from_rosbag(bag, imgtopic, H=180, W=240):
    imgs = []
    
    progress_bar = tqdm.tqdm(total=bag.get_message_count(imgtopic))
    for topic, msg, t in bag.read_messages(imgtopic):
        img_str = str(msg)
        img_str = img_str[img_str.find("data")+6:]
        img_str = img_str[1:-1].split(',')
        pixel_values = [int(v) for v in img_str]
        image_array = np.array(pixel_values, dtype=np.uint8)
        image_array = image_array.reshape((msg.height, msg.width,3))

        # 图片的高度和宽度是否和bag文件中的一致
        if abs(H- msg.height) > 2 or abs(W-msg.width) > 2:
            print(f"WARNING: H, W mismatch: {msg.height}, {msg.width}, and resize to {H}, {W}") 
            image_array = cv2.resize(image_array, (W, H)) #不一致就必须要resize  

        imgs.append(image_array)
        progress_bar.update(1)

        # if abs(H- msg.height) > 2 or abs(W-msg.width) > 2:
        #     print(f"WARNING: H, W mismatch: {msg.height}, {msg.width}, {H}, {W}")    

    return imgs


def read_tss_us_from_rosbag(bag, imgtopic):
    tss_us = []
    for topic, msg, t in bag.read_messages(imgtopic):
        tss_us.append(msg.header.stamp.to_nsec() / 1e3)
    return tss_us


def read_poses_from_rosbag(bag, posestopic, T_marker_cam0, T_cam0_cam1):
    progress_bar = tqdm.tqdm(total=bag.get_message_count(posestopic))

    poses = []
    tss_us_gt = []
    for topic, msg, t in bag.read_messages(posestopic):
        if msg._type == "nav_msgs/Odometry":
            ps = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        else:
            ps = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        T_world_marker = np.eye(4)
        T_world_marker[:3, 3] = ps[:3]
        T_world_marker[:3, :3] = R.from_quat(ps[3:]).as_matrix()
        
        T_world_cam = T_world_marker @ T_marker_cam0
        T_world_cam = T_world_cam @ T_cam0_cam1

        T_world_cam = np.concatenate((T_world_cam[:3, 3], R.from_matrix(T_world_cam[:3, :3]).as_quat()))
        poses.append(T_world_cam)

        tss_us_gt.append(msg.header.stamp.to_nsec() / 1e3)
                     
        progress_bar.update(1)
    return np.array(poses), tss_us_gt

# 读取IMU数据
def read_imu_from_rosbag(bag, imutopic):
    progress_bar = tqdm.tqdm(total=bag.get_message_count(imutopic))

    all_imu = []
    
    for topic, msg, t in bag.read_messages(imutopic):
        acc_x = msg.linear_acceleration.x
        acc_y = msg.linear_acceleration.y
        acc_z = msg.linear_acceleration.z
        w_x   = msg.angular_velocity.x
        w_y   = msg.angular_velocity.y
        w_z   = msg.angular_velocity.z
        # timeimu = "%.0f" % (msg.header.stamp.to_sec()*1000000000) 
        timeimu = msg.header.stamp.to_nsec()
        all_imu.append([timeimu,w_x,w_y,w_z,acc_x,acc_y,acc_z])  
                     
        progress_bar.update(1)
    return all_imu

#获取ns时间戳
def read_tss_ns_from_rosbag(bag, imgtopic):
    tss_us = []
    for topic, msg, t in bag.read_messages(imgtopic):
        tss_us.append(msg.header.stamp.to_nsec())
    return tss_us

def read_calib_from_bag(bag, imtopic):
    for topic, msg, t in bag.read_messages(imtopic):
        K = msg.K
        break
    return K


def read_t0us_evs_from_rosbag(bag, evtopic, t0us_start=0):
    has_read = False
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            t0_us = ev.ts.to_nsec() / 1e3
            if t0_us >= t0us_start:
                has_read = True
                break
        if has_read == True:
            break
    return t0_us