import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import random
import facenet
from scipy import misc
import sklearn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
dist = []

pic_store= 'picture'  # "Points to a module containing the definition of the inference graph.")
image_size=160 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."

images = None

frame_interval=5 # frame intervals
def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    tmp_image_paths = []
    img_list = []

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, 'model_check_point/')
    
    if (os.path.isdir(image_paths)):
        for item in os.listdir(image_paths):
            tmp_image_paths.insert(0,pjoin('picture',item))
    else:
        tmp_image_paths=copy.copy(image_paths)

    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    # images = np.stack(img_list)
    return img_list,tmp_image_paths,pnet, rnet, onet


#restore facenet model
print('建立Real-time face recognize模型')
# Get input and output tensors、
with tf.Graph().as_default():
        with tf.Session() as sess:

            print('载入人脸库>>>>>>>>')
            images_tmp,tmp_image_paths,pnet, rnet, onet = load_and_align_data(pic_store,160,44,1.0)
            nrof_images = len(tmp_image_paths)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, tmp_image_paths[i]))
            print('')

            print('开始加载模型')
            # Load the model
            model_checkpoint_path='model_check_point/20180720'
            facenet.load_model(model_checkpoint_path)
            
            print('建立facenet embedding模型')
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            #obtaining frames from camera--->converting to gray--->converting to rgb
            #--->detecting faces---->croping faces--->embedding--->classifying--->print
            video_capture = cv2.VideoCapture(0)
            # video_capture.set(3,1440)
            # video_capture.set(4,1080)
            c=0            
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()              
                timeF = frame_interval
                               
                if(c%timeF == 0): #frame_interval==3, face detection every 3 frames                    
                    find_results=[]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                        
                    if gray.ndim == 2:
                        img = to_rgb(gray)
                                            
                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                   
                    if len(bounding_boxes) < 1:
                        continue
                    else:
                        img_size = np.asarray(frame.shape)[0:2]
                        det = np.squeeze(bounding_boxes[0,0:4])
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-44/2, 0)
                        bb[1] = np.maximum(det[1]-44/2, 0)
                        bb[2] = np.minimum(det[2]+44/2, img_size[1])
                        bb[3] = np.minimum(det[3]+44/2, img_size[0])
                        cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]

                    nrof_faces = bounding_boxes.shape[0]#number of faces
                    print('找到人脸数目为：{}'.format(nrof_faces))
                    

                    for face_position in bounding_boxes:                        
                        face_position=face_position.astype(int)                       
                        print((int(face_position[0]), int( face_position[1])))
                        #word_position.append((int(face_position[0]), int( face_position[1])))
                    
                        images=cv2.rectangle(frame, (face_position[0], 
                                        face_position[1]), 
                                (face_position[2], face_position[3]), 
                                (0, 255, 0), 2)
                        
                        # cropped=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
                        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')               
                        prewhitened = facenet.prewhiten(aligned)                 
                        images_tmp.append(prewhitened) 
                        print(len(images_tmp))  
                        image = np.stack(images_tmp)                    
                        emb_data = sess.run(embeddings, 
                                            feed_dict={images_placeholder: image, 
                                                    phase_train_placeholder: False }) 
                        images_tmp.pop()                    
                        for i in range(len(emb_data)-1):
                            dist.append(np.sqrt(np.sum(np.square(np.subtract(emb_data[3,:], emb_data[i,:])))))
                            
                        a = dist.index(min(dist))  
                        # print(images_tmp[a])
                        # print(os.path.basename(images_tmp[a]))
                        name = os.path.splitext(os.path.basename(tmp_image_paths[a]))[0]
                        print(name) 
                        dist = [] 
                            
                    # cv2.putText(frame,'detect:{}'.format(find_results), (50,100), 
                    #         cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
                    #         thickness = 2, lineType = 2)                             
                        # print(faces)
                c+=1
                # Draw a rectangle around the faces                
                # Display the resulting frame
                if nrof_faces >= 1 :
                    cv2.imshow('Video', images)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
# When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()

