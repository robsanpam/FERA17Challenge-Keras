from mpi4py import MPI
import os
import csv
import cv2
import numpy as np
import sys
import time
from six.moves import cPickle as pickle
import shutil

def get_csv_id(filename):
    """Function that returns the name of the 
    csv file with the corresponding labels for
    a given video file.
      Args:
        x: String name of a video.
      Returns:
        A string csv file name for x.
    """
    return filename[:-6] + ".csv"


def modify_image(image, target_w, target_h, grayscale=False):
    """Function that performs some image preprocessing
    operations.
      Args:
        image: OpenCV frame array that will be processed.
        target_h: Integer height to which the video will be
            resized.
        target_w: Integer width to which the image will be
            resized.
        grayscale: Boolean to indicate whether to convert the 
            image into grayscale.
      Returns:
        Numpy array of modified image.
    """
    
    mod = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if grayscale:
        mod = cv2.cvtColor(mod, cv2.COLOR_BGR2GRAY)
    return np.expand_dims(np.array(mod, dtype=np.float32), axis=0)


def broadcastList(files, jobs, offset=0):
    """Function that returns a dictionary with
    rank numbers as keys and list of jobs
    as values. Each job is defined as an array
    that contains the start and end index of the
    list of videos to be processed by the 
    specified rank number, 
    i.e. {0: [14, 20], ...}.
    The function will first assign the same number
    of files as the first job to each rank. Then, 
    the function will be called by itself to
    assign the remaining items as a new job to
    each rank, starting from rank 0,
    i.e. {0: [[14,20], [220,226]], ...}. The
    function adapts to the number of threads
    available at runtime.
      Args:
        files: String list of videos to be
            divided per rank.
        jobs: Dictionary of current job
            assignments per rank.
        offset: Integer number offset for
            job assignation when called
            recursively.
      Returns:
        A dictionary of job assignments per rank.
    """
    
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
    

def process_information(jobs, video_list, video_dir, label_dir, result_dir,
                       dict_dir, target_h, target_w, skips=0, pack=3, 
                       grayscale=False, label_index=0):
    """Function that creates a dictionary of file locations and
    labels to be used with a modified Keras ImageDataGenerator
    class that allows multi-label learning. This function also
    samples a video file into appended contiguous frames with a
    skips number of frames that are omitted between samples. 
    Frames are resized to (target_w, target_h) and saved to result_dir.   
      Args:
        jobs: Dictionary of job assignments per rank.
    video_list: String list of videos to be processed. They
            should be inside the directory video_dir.
    video_dir: String path to where video_list videos are
        located.
    label_dir: String path to where the csv label files are
        located.
    result_dir: String path to where the video frames will 
        be saved.
    dict_dir: String path to where the train and validation
        dictionaries will be saved.
    target_h: Integer height to which each video frame will be
        resized.
    target_w: Integer width to which each video frame will be
        resized.
    skips: Integer number of frames to skip after saving a
        video frame.
    pack: Integer number of images to append to single 
        instance.
    grayscale: Boolean to indicate whether to convert the 
        image into grayscale.
    label_index: Integer number of frame in the pack of 
        images whose label will be assigned to the whole
        pack of images.
    """

    if isinstance(jobs, list):
        jobs = [jobs]
        
    for limits in jobs:
        files = video_list[limits[0]:limits[1]]
        dictionary = {}
        
        for filename in files:
            y = np.zeros((0, 10))
            X = np.zeros((0, target_h, target_w, 3))
            vid_path = os.path.join(video_dir, filename)
            csv_path = os.path.join(label_dir, get_csv_id(filename))

            # Capture frames from video
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert pack <= total_frames
            exit = False
            instances = 0
            frames_count = 0
            while (cap.isOpened() and not exit):
                ret, frame = cap.read()
                frames_count += 1
                if ret != False:
                    # Merge pack if there are enough images to read
                    if frames_count + (pack-1) <= total_frames:
                        merged = modify_image(frame, target_w, target_h)
                        for _ in range(pack-1):
                            r, frame2 = cap.read()
                            frames_count += 1
                            merged = np.append(merged, 
                                               modify_image(frame2, target_w, target_h),
                                               axis=0)

                        # Create image path
                        im_path = result_dir+filename.split(".")[0]+"_%05d"%(instances)+".pickle"

                        # Path of image generator
                        dict_key = result_dir.split("/")[-2:-1][0]+"/"+filename.split(".")[0]+"_%05d"%(instances)+".pickle"
                        
                        # Get label for instance
                        label = []
                        with open(csv_path, newline='') as csvfile:
                            reader = csv.reader(csvfile)
                            label = list(map(int,list(reader)
                                             [frames_count-pack+label_index]))[1:]
                        
                        # Save packed image
                        try:
                            f = open(im_path, 'wb')
                            pickle.dump(merged, f, pickle.HIGHEST_PROTOCOL)
                            f.close()
                        except Exception as e:
                            print('Unable to save data to',
                                  im_path, ':', e)
                            raise
                        
                        # Include instance and label to dictionary
                        dictionary[dict_key] = label
                  
                    # Skip frames if selected
                    for skip in range(skips):
                        cap.read()
                        frames_count += 1
                    instances += 1
                else:
                    exit = True
            cap.release()
        

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

def create_dataset(orig_db_path="~", # Path to original db
                   main_dir="~", # Location of new dataset
                   target_h=112, target_w=112,
                   view=6, skips=50, pack=16,
                   grayscale=False, label_index=8):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set hyperparameters
    target_h = 112
    target_w = 112
    view = 6 # Number from 1 to 9 or -1 for all views
    skips = 50
    pack = 16
    grayscale = False
    label_index = 8

    # Name of directory for new dataset
    dataset_id = 'FERA17DB_3D-h'+str(target_h)+'-w'+str(target_w)+'-v'+str(view)+'-s'+str(skips)+'-p'+str(pack)+'-gs'+str(grayscale)[0]+'-li'+str(label_index)+'/'

    # Defining input files' paths
    # Names are the default ones from the original FERA17 db
    train_video_dir = orig_db_path+"FERA17_Train_MV/"
    valid_video_dir = orig_db_path+"FERA17_Valid_MV/"
    train_label_dir = orig_db_path+"Train_Occ/"
    valid_label_dir = orig_db_path+"Valid_Occ/"

    # Defining output files' paths
    main_dir = main_dir+dataset_id
    au_list = ["AU1", "AU4", "AU6", "AU7", "AU10", "AU12", "AU14", "AU15", "AU17", "AU23"]
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
        print("Creating training set.")
    for worker, job in jobs_train.items():
        if rank == int(worker):
            process_information(job,
                                training_vid_list,
                                train_video_dir,
                                train_label_dir,
                                train_imgs_dir,
                                train_dict_dir,
                                target_h,
                                target_w,
                                skips,
                                pack,
                                grayscale,
                                label_index)
            break

    comm.Barrier()

    if rank == 0:
        print("Done.\n")
        print("Creating validation set.")

    # Create validation set
    for worker, job in jobs_valid.items():
        if rank == int(worker):
            process_informationx(job,
                                 valid_vid_list,
                                 valid_video_dir,
                                 valid_label_dir,
                                 valid_imgs_dir,
                                 valid_dict_dir,
                                 target_h,
                                 target_w,
                                 skips,
                                 pack,
                                 grayscale,
                                 label_index)
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

        # Create a classes dictionary
        classes = {
            "AU1": 0,
            "AU4": 1,
            "AU6": 2,
            "AU7": 3,
            "AU10": 4,
            "AU12": 5,
            "AU14": 6,
            "AU15": 7,
            "AU17": 8,
            "AU23": 9
        }

        try:
            f = open(dictionary_dir + "classes_dict.p", 'wb')
            pickle.dump(classes, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', dictionary_dir + "classes_dict.p:", e)
            raise
        print("Done.\n")

        # Remove temporal directories created
        if os.path.exists(train_dict_dir):
            shutil.rmtree(train_dict_dir)
        if os.path.exists(valid_dict_dir):
            shutil.rmtree(valid_dict_dir)

        print("Duration %.1f minutes.\n" % ((time.time() - starttime)/60))


def main(argv):
    create_dataset(orig_db_path=str(argv[0]),
                   main_dir=str(argv[1]),
                   target_h=str(argv[2]), 
                   target_w=str(argv[3]),
                   view=str(argv[4]),
                   skips=str(argv[5]), 
                   pack=str(argv[6]),
                   grayscale=str(argv[7]),
                   label_index=str(argv[8]))

if __name__ == "__main__":
    main(sys.argv[1:])    
