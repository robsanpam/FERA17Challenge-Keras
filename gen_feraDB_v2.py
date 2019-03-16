from mpi4py import MPI
# from tensorflow.python.platform import gfile
# from tensorflow.python.platform import app
import os
import csv
import cv2
import numpy as np
import sys
import math
import time
from six.moves import cPickle as pickle
import shutil
from PIL import Image as pil_image

def get_csv_id(filename):
    return filename[:-6] + ".csv"


def broadcastList(files, jobs, offset=0):
   
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    step_size = 0
    
    # When list size is less than number of threads
    if size >= len(files):
        used_ranks = np.arange(0,len(files))
        step_size = 1
        for rank in used_ranks:
            start = rank
            stop = rank+1
            if rank in jobs.keys():
                jobs[rank] = np.append([jobs[rank]], [[start+offset,stop+offset]], axis=0)
            else:
                jobs[rank] = [start,stop]
    else: 
        step_size = len(files)//size
        used_ranks = np.arange(0, len(files)//step_size)
        for rank in used_ranks:
            start = max(0,rank*step_size-1)
            stop = rank*step_size-1+step_size
            if rank in jobs.keys():
                jobs[rank] = np.append([jobs[rank]], [[start+offset,stop+offset]], axis=0)
            else:
                jobs[rank] = [start,stop]

    # If there are pending files to assign
    if step_size*max(used_ranks)+offset-1+step_size <= len(files)-1:
        broadcastList(files[step_size*max(used_ranks)+offset-1+step_size:], jobs, step_size*max(used_ranks)+offset-1+step_size)

    return jobs


def get_video_capture_and_frame_count(path):
    assert os.path.isfile(
      path), "Couldn't find video file:" + path + ". Skipping video."
    cap = None
    
    if path:
        cap = cv2.VideoCapture(path)

    assert cap is not None, "Couldn't load video capture:" + path + ". Skipping video."

    # compute meta data of video
    if hasattr(cv2, 'cv'):
        frame_count = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, frame_count


def get_next_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return np.array(frame, dtype=np.float32)

def compute_dense_optical_flow(prev_image, current_image):
    old_shape = current_image.shape
    prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
    current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
    assert current_image.shape == old_shape
    hsv = np.zeros_like(prev_image)
    hsv[..., 1] = 255
    flow = None
    flow = cv2.calcOpticalFlowFarneback(prev=prev_image_gray,
                                        next=current_image_gray, flow=flow,
                                        pyr_scale=0.8, levels=15, winsize=5,
                                        iterations=10, poly_n=5, poly_sigma=0,
                                        flags=10)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)



def repeat_image_retrieval(cap, file_path, video, take_all_frames, steps, frame,
                           prev_frame_none, frames_counter):
    stop = False

    if frame and prev_frame_none or steps <= 0:
        stop = True
        return stop, cap, video, steps, prev_frame_none, frames_counter

    if not take_all_frames:
        # repeat with smaller step size
        steps -= 1

    prev_frame_none = True
    print("reducing step size due to error for video: ", file_path)
    frames_counter = 0
    cap.release()
    cap = get_video_capture_and_frame_count(file_path)
    # wait for image retrieval to be ready
    time.sleep(2)

    return stop, cap, video, steps, prev_frame_none, frames_counter


def video_file_to_ndarray(file_path, n_frames_per_video, height, width,
                          dense_optical_flow, n_channels=3):
    
    cap, frame_count = get_video_capture_and_frame_count(file_path)

    take_all_frames = False
    # if not all frames are to be used, we have to skip some -> set step size accordingly
    if n_frames_per_video == 'all':
        take_all_frames = True
        video = np.zeros((frame_count, height, width, n_channels), dtype=np.uint8)
        steps = frame_count
        n_frames = frame_count
    else:
        video = np.zeros((n_frames_per_video, height, width, n_channels),
                         dtype=np.uint8)
        steps = int(math.floor(frame_count / n_frames_per_video))
        n_frames = n_frames_per_video

    assert not (frame_count < 1 or steps < 1), str(file_path) + " does not have enough frames. Skipping video."

    image = np.zeros((height, width, n_channels),
                     dtype=np.uint8)
    
    frames_counter = 0
    prev_frame_none = False
    restart = True
    image_prev = None

    while restart:
        for f in range(frame_count):
            if math.floor(f % steps) == 0 or take_all_frames:
                frame = get_next_frame(cap)
                # unfortunately opencv uses bgr color format as default
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # special case handling: opencv's frame count sometimes differs from real frame count -> repeat
                if frame is None and frames_counter < n_frames:
                    stop, cap, steps, prev_frame_none, frames_counter = repeat_image_retrieval(cap, file_path, take_all_frames, steps, frame, prev_frame_none,frames_counter)
                    if stop:
                        restart = False
                        break
                    else:
                        video.fill(0)
                else:
                    if frames_counter >= n_frames-1:
                        restart = False
                        break

                image[:, :, :] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                if dense_optical_flow:
                    if image_prev is not None:
                        frame_flow = compute_dense_optical_flow(image_prev, image)
                    else:
                        frame_flow = np.zeros((height, width, 3))
                    
                    # saving current frame for next iteration    
                    image_prev = image.copy()

                    # TODO: saving a concatenation of OF + RGB
                    if False:
                        image_with_flow = np.append(image.copy(), frame_flow, axis=2)
                    else:
                        video[frames_counter, :, :, :] = frame_flow

                else:
                    video[frames_counter, :, :, :] = image
                
                frames_counter += 1

            else:
                get_next_frame(cap)
                
    v = video.copy()
    cap.release()
    return video
    

def process_info(jobs, qty_labels, video_list,
                 video_dir, label_dir, result_dir,
                 dict_dir, target_h, target_w, 
                 dense_optical_flow, n_channels=3,
                 rescale=1/255.):    
    
    if isinstance(jobs, list):
        jobs = [jobs]

    for limits in jobs:
        files = video_list[limits[0]:limits[1]]
        dictionary = {}

        for filename in files:

            # GET WHOLE VIDEO
            vid_path = os.path.join(video_dir, filename)

            try:
                v = video_file_to_ndarray(file_path=vid_path,
                            n_frames_per_video='all',
                            height=target_h, width=target_w,
                            dense_optical_flow=dense_optical_flow)

            except Exception as e:
                print(e)

            # GET WHOLE LABEL
            y = np.zeros((0, qty_labels))
            csv_path = os.path.join(label_dir, get_csv_id(filename))

            with open(csv_path, newline='') as csvfile:
                labels = csv.reader(csvfile)
                # Delete header and row index from reader
                labels = np.delete(list(labels)[1:], 0, axis=1).astype(np.int)
                # Delete AUs not considered for the thesis
                labels = np.delete(labels, [0,1,7,8,9], axis=1)

            # v = v.astype(np.float)

            for instances, data in enumerate(zip(v, labels)):
                
                label = data[1]
                im = data[0]
                
                # Sample-wise rescale
                # im *= rescale
                # print(im.shape)
                
                # Image path
                im_path = result_dir+filename.split(".")[0]+"_%05d"%(instances)+".jpg"
                # Image path in dictionary
                dict_key = result_dir.split("/")[-2:-1][0]+"/"+filename.split(".")[0]+"_%05d"%(instances)+".jpg"
                
                # Converto to BGR to save it correctly
                im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                
                # Save BGR image
                cv2.imwrite(im_path, im_bgr)
                
                # Include instance and label to dictionary    
                dictionary[dict_key] = label
               
            # Save dictionary
            try:
                f = open(dict_dir + str(limits[0]) + "_" + str(limits[1]) + 
                         ".pickle", 'wb')
                pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
                f.close()
            except Exception as e:
                print('Unable to save data to',
                      dict_dir + str(limits[0]) + "_" + str(limits[1]) + 
                      ".pickle", ':', e)
                raise

def create_dataset(orig_db_path, main_dir,
                   target_h, target_w,
                   view, dense_optical_flow):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Name of directory for new dataset
    dataset_id = 'FERA17DB-OF'+str(dense_optical_flow)[0]+'-h'+str(target_h)+'-w'+str(target_w)+'-v'+str(view)+'/'

    # Defining input files' paths
    # Names are the default ones from the original FERA17 db
    train_video_dir = orig_db_path+"FERA17_Train_MV/"
    valid_video_dir = orig_db_path+"FERA17_Valid_MV/"
    train_label_dir = orig_db_path+"Train_Occ/"
    valid_label_dir = orig_db_path+"Valid_Occ/"

    # Defining output files' paths
    main_dir = main_dir+dataset_id
    au_list = ["AU6", "AU7", "AU10", "AU12", "AU14"]

    qty_labels = len(au_list)
    images_dir = main_dir + "images/"
    train_dir = images_dir + "train/"
    valid_dir = images_dir + "valid/"
    train_imgs_dir = train_dir + str(
        au_list[0]) + "/"  # Folder where every training image will be saved
    valid_imgs_dir = valid_dir + str(
        au_list[0]) + "/"  # Folder where every validation image will be saved
    dictionary_dir = main_dir + "meta/"
    train_dict_dir = dictionary_dir + "dicts_train/"
    valid_dict_dir = dictionary_dir + "dicts_valid/"

    # Create required directories
    if rank == 0:
        try:
            assert view in np.delete(np.arange(-1,10),1)
        except AssertionError as e:
            e.args += ('View must be a number between 1-9 or -1 for all views.')
            raise
        starttime = time.time()    
        print("\nCreating folders...")
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)

        for au in au_list:
            if not os.path.exists(train_dir + au):
                os.makedirs(train_dir + au)
            if not os.path.exists(valid_dir + au):
                os.makedirs(valid_dir + au)

        if not os.path.exists(train_dict_dir):
            os.makedirs(train_dict_dir)
        if not os.path.exists(valid_dict_dir):
            os.makedirs(valid_dict_dir)
        print("Done.\n")

    comm.Barrier()

    # Filter training and validation videos according to view
    if view != -1:
        training_vid_list = [file for file in sorted(os.listdir(train_video_dir)) if int(file[-5]) == view]
        valid_vid_list = [file for file in sorted(os.listdir(valid_video_dir)) if int(file[-5]) == view]
    else: 
        training_vid_list = [file for file in sorted(os.listdir(train_video_dir))]
        valid_vid_list = [file for file in sorted(os.listdir(valid_video_dir))]

    # Create jobs for each rank and broadcast them
    jobs_train = {}
    jobs_valid = {}

    if rank == 0:
        jobs_train = broadcastList(training_vid_list, jobs_train)
        jobs_valid = broadcastList(valid_vid_list, jobs_valid)
    else:
        jobs_train = None
        jobs_valid = None

    comm.Barrier()

    jobs_train = comm.bcast(jobs_train, root=0)
    jobs_valid = comm.bcast(jobs_valid, root=0)        
        
    # Create training set
    if rank == 0:
        print('-'*40)
        print("Creating training set")
    for worker, job in jobs_train.items():
        if rank == int(worker):
            process_info(job, qty_labels,
                           training_vid_list,
                           train_video_dir,
                           train_label_dir,
                           train_imgs_dir,
                           train_dict_dir,
                           target_h,
                           target_w,
                           dense_optical_flow=dense_optical_flow)           
            break

    comm.Barrier()

    if rank == 0:
        print("Done.\n")
        print("Creating validation set")

    # Create validation set
    for worker, job in jobs_valid.items():
        if rank == int(worker):
            process_info(job, qty_labels,
                           valid_vid_list,
                           valid_video_dir,
                           valid_label_dir,
                           valid_imgs_dir,
                           valid_dict_dir,
                           target_h,
                           target_w,
                           dense_optical_flow=dense_optical_flow)
            break

    comm.Barrier()
    
    if rank == 0:
        print("Done.\n")
        print("Finishing up...")

        # Merge train and validation dictionaries
        files_valid = sorted(os.listdir(valid_dict_dir))
        files_train = sorted(os.listdir(train_dict_dir))

        valid_dict = {}
        for filename in files_valid:
            temp_dict = pickle.load(open(valid_dict_dir + filename, 'rb'))
            valid_dict = {**valid_dict, **temp_dict}

        train_dict = {}
        for filename in files_train:
            temp_dict = pickle.load(open(train_dict_dir + filename, 'rb'))
            train_dict = {**train_dict, **temp_dict}

        all_y_labels = {**train_dict, **valid_dict}

        try:
            f = open(dictionary_dir + "all_y_labels.p", 'wb')
            pickle.dump(all_y_labels, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', dictionary_dir + "all_y_labels.p:", e)
            raise    
        
        # Remove temporal directories created
        if os.path.exists(train_dict_dir):
            shutil.rmtree(train_dict_dir)
        if os.path.exists(valid_dict_dir):
            shutil.rmtree(valid_dict_dir)
        
        # Create a classes dictionary
        classes = {
            "AU6": 0,
            "AU7": 1,
           "AU10": 2,
           "AU12": 3,
           "AU14": 4,
        }

        try:
            f = open(dictionary_dir + "classes_dict.p", 'wb')
            pickle.dump(classes, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', dictionary_dir + "classes_dict.p:", e)
            raise
        print("Done.\n")

        print("Duration %.1f minutes.\n" % ((time.time() - starttime)/60))

def main():
    create_dataset(orig_db_path="/home/mcc/robsanpam/db/FERA17/", 
                   main_dir="/home/mcc/robsanpam/db/",
                   target_h=224, target_w=224,
                   view=6, dense_optical_flow=True)

if __name__ == "__main__":
    main()
