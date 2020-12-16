from imutils import face_utils
import dlib
import imutils
import cv2
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("--------- Python OpenCV Tutorial ---------")
capture = cv2.VideoCapture("zoom_1.mp4")
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)


for i in range(250):
    ret, image = capture.read()

class node:
    def __init__(self,x=None,y=None,id=None):
        self.id=id
        self.x=x
        self.y=y
        self.next=None
        self.queue = Queue(maxsize=10)

class linked_list:
    def __init__(self):
        self.head = node()

    def append(self,x,y,id):
        new_node = node(x,y,id)
        cur = self.head
        while cur.next!=None:
            cur = cur.next
        cur.next = new_node
    
    def length(self):
        cur = self.head
        total = 0
        while cur.next!=None:
            total+=1
            cur = cur.next
        return total
    
    def search(self,x,y):
        if self.length() == 0:
            return None
        cur_node = self.head
        while cur_node.next!=None:
            cur_node=cur_node.next
            if ((x>(cur_node.x-10)) and (x<(cur_node.x+10))):
                if ((y>(cur_node.y-10)) and (y<(cur_node.y+10))):
                    return cur_node
        return None

    def display(self):
        elems = []
        cur_node = self.head
        while cur_node.next!=None:
            cur_node = cur_node.next
            elems.append(cur_node.id)
        print(elems)

my_list = linked_list()
#5 elements in lips_distance:minimal, max, average, # of data in queue(at most 10), # of data over average
#support 20 faces at most
lips_distance=np.zeros((20,5))
lips_distance_plot=np.zeros((1000))
x_plot=np.arange(0, 100, 1) 
x_index=0
talking_index=0
n_frame=0


total_students=100
attendance=0

while(True):
    for i_i in range(1):
        ret, image = capture.read()
    image=cv2.resize(image, (int(np.shape(image)[1]/2), int(np.shape(image)[0]/2)), interpolation = cv2.INTER_AREA)
    n_frame=n_frame+1
    #image=image[:,int(np.shape(image)[0]*3/4):,:]
    #print(np.shape(image))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # enumerate()方法用于将一个可遍历的数据对象(列表、元组、字典)组合
    # 为一个索引序列，同时列出 数据下标 和 数据 ，一般用在for循环中
    mouse=np.zeros((len(rects),2))
    distance=np.zeros((100,2))
    for(i, rect) in enumerate(rects):
        shape = predictor(gray, rect)  # 标记人脸中的68个landmark点
        shape = face_utils.shape_to_np(shape)  # shape转换成68个坐标点矩阵
    
        (x, y, w, h) = face_utils.rect_to_bb(rect)  # 返回人脸框的左上角坐标和矩形框的尺寸
        res = my_list.search(x,y)
        if res == None:
            id = my_list.length() + 1
            my_list.append(x,y,id)
            res = my_list.search(x,y)
            print("detect a new face")
        else:
            id = res.id
            res.x = x
            res.y = y
            print("find an old face")

        #my_list.display()

        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        landmarksNum = 0;
        #print(shape)
        ##coordinate x for point 52
        #mouse[i,0]=shape[51,0]
        ##coordinate y for point 52
        #mouse[i,1]=shape[51,1]

        ##coordinate x for point 58
        #mouse[i,0]=shape[57,0]
        ##coordinate y for point 58
        #mouse[i,1]=shape[57,1]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #眼睛左右距离
        eye_width = shape[39,0] - shape[36,0]

        #眼睛上下距离
        eye_distance = shape[40,1] - shape[38,1]
        
        #嘴唇左右距离
        lip_width = shape[54,0] - shape[48,0]

        #嘴唇上下距离
        lip_distance = shape[57,1] - shape[51,1]
        #if lip_distance>3.5:
        #    talking_index=talking_index+1        
        if res.id==2:
            lips_distance_plot[x_index] = lip_distance
            x_index=x_index+1
            #if lip_distance>3.5:
            #    talking_index=talking_index+1
            plt.plot(lips_distance_plot[:x_index])
            #plt.text(0.5, 1, 'put some text')
            #plt.show()
            plt.pause(0.01)        
        #print(res.id)
        #print(lips_distance_plot[x_index])

        #update minial lips distance
        if lips_distance[res.id, 0]== 0:
            print("update minial lips distance")
            lips_distance[res.id,0] = lip_distance
            print("initial minial lips distance for ",res.id," is ",lips_distance[res.id,0])
        else:
            if lips_distance[res.id,0] > lip_distance:
                lips_distance[res.id,0] = lip_distance
                print("update minial lips distance for ",res.id," is ",lips_distance[res.id,0])

        #update maximum lips distance
        if lips_distance[res.id, 1]== 0:
            print("update maximum lips distance")
            lips_distance[res.id,1] = lip_distance 
            print("initial maximul lips distance for ",res.id," is ",lips_distance[res.id,0])
        else:
            if lips_distance[res.id,1] < lip_distance:
                lips_distance[res.id,1] = lip_distance 
                print("update maximul lips distance for ",res.id," is ",lips_distance[res.id,0])
         
        #average of lips distance
        lips_distance[res.id,2] = (lips_distance[res.id,0] + lips_distance[res.id,1])/2

        res.queue.put(lip_distance - lips_distance[res.id,2])
        lips_distance[res.id,3]+=1
        if lip_distance > lips_distance[res.id,2]:
            lips_distance[res.id,4]+=1 
        if lips_distance[res.id,3] == 10:
            lips_distance[res.id,3]-=1
            if res.queue.get() > 0:
                lips_distance[res.id,4]-=1
            print("decide status")
            #silent
            if lips_distance[res.id,4] < 4:
                status = 0
                cv2.putText(image, "Silent #{}".format(res.id), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #smile
            elif lips_distance[res.id,4] >= 8:
                status = 1
                cv2.putText(image, "Smile #{}".format(res.id), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #talk
            else:
                status = 2
                talking_index=talking_index+2                
                cv2.putText(image, "Talk #{}".format(res.id), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
        #print(lips_distance)
                

        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            # cv2.putText(image, "{}".format(landmarksNum), (x, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
            # landmarksNum = landmarksNum + 1;
        landmarksNum = 0;

    #attendance=
    cv2.putText(image, "Total Students #{}".format(5), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)
    cv2.putText(image, "Attendance #{}".format(i+1), (10, 30*2),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)    
    cv2.putText(image, "Participation Ratio #{}".format((i+1)/5.0), (10, 30*3),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)
    cv2.putText(image, "Talking Time/Total Time Ratio #{}".format(talking_index/(n_frame*(i+1))), (10, 30*4),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)
    if talking_index/(n_frame*(i+1))>0.50:    
        cv2.putText(image, "Class Quality : Good", (10, 30*5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)
    elif talking_index/(n_frame*(i+1))>0.3:
        cv2.putText(image, "Class Quality : Medium", (10, 30*5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)
    else:
        cv2.putText(image, "Class Quality : Bad", (10, 30*5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255), 2)            
    #print(mouse)
    # same # matplotlib
    #plot(distance)
    cv2.imshow("Output", image)    
    #cv2.imshow("result", frame)
    c = cv2.waitKey(10)
    if c == 27: # ESC
        break
capture.release()
cv2.waitKey(0) 
cv2.destroyAllWindows()

