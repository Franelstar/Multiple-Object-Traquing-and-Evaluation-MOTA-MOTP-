import cv2 as cv
import sys
import os
import numpy as np
import time
sys.path
sys.path.append('./yolo')
#import darknet

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
    
def get_output_layers(net):    
        layer_names = net.getLayerNames()        
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

class DetectorYolo3_GPU:
    def __init__(self,configPath,weightPath,metaPath):        
        self.configPath = configPath
        self.weightPath = weightPath
        self.metaPath = metaPath        
        self.metaMain=None
        self.netMain=None
        self.altNames=None
        self.load_model()

    def load_model(self):        
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
   

    def predict(self,frame, classname, thresh=0.5, nms=0.5,min_W=16,min_H=32):          
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        im,_ = darknet.array_to_image(frame)      
        detections = darknet.detect_image(self.netMain, self.metaMain, im, thresh, nms)  
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # darknet_image = darknet.make_image(frame.shape[1],frame.shape[0],3)        
        # darknet.copy_image_from_bytes(darknet_image,frame.tobytes())    
        # detections = darknet.detect_image(self.netMain, self.metaMain, darknet_image, thresh,nms)   
        boxes=[]
        for detection in detections:
            if classname !=None and detection[0].decode()!=classname:
                continue
            # if detection[1] < 0.9:
            #     continue
            c_x, c_y, w, h = detection[2][0], detection[2][1],detection[2][2],detection[2][3] 
            # if w<min_W or h<min_H:
            #     continue
            xmin, ymin, xmax, ymax = convertBack(float(c_x), float(c_y), float(w), float(h))            
            boxes.append([xmin, ymin, xmax, ymax])
        return boxes

class DetectorYolo3_CPU:
    def __init__(self, cfg, weights,classFile):
        self._classFile=classFile
        self._weights=weights
        self._cfg = cfg        
        self.load_model()
    def load_model(self):
        with open(self._classFile, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]        
        self.net = cv.dnn.readNetFromDarknet(self._cfg,self._weights)
    def predict(self,frame, classname,thresh=0.5, nms=0.5,min_W=16,min_H=32):        
        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392
        blob = cv.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(get_output_layers(self.net)) 
        #class_ids = []
        confidences = []
        boxes = []       
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if self.classes[class_id]!=classname:
                    continue
                confidence = scores[class_id]
                if confidence > thresh:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2   
                    # class_ids.append(class_id)
                    confidences.append(float(confidence))               
                    boxes.append([x, y, w, h])          
        indices = cv.dnn.NMSBoxes(boxes, confidences, thresh, nms)
        dets=[]
        for i in indices:            
            box = boxes[i[0]]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            dets.append([int(x),int(y),int(x+w),int(y+h)])
        return dets

### For testing this class
detectIter=5	# Re-detect after detectIter frame
iouThr=0.4		# To determine the same object
movThr=3		# To determine moving or not (pixel)
confThr=0.15 	# Object or not
nmsThr=0.3		# for nms

## Gennerate clolor list
vs=[0,85,170,255]
colors=[]
for v1 in vs:
	for v2 in vs:
		for v3 in vs:
			colors.append((v1,v2,v3)) 

# Draw boxes of detection
def DrawBoxes(detections, img):
	for detection in detections:		
		xmin, ymin, xmax, ymax = detection		
		pt1 = (xmin, ymin)
		pt2 = (xmax, ymax)
		cv.rectangle(img, pt1, pt2, (0, 255, 0), 1)		
	return img

def Process():
    #vin=None
    #vin = "rtsp://192.168.0.10:554/user=admin_password=_channel=1_stream=0.sdp"
    vin="/home/Downloads/viethq/fight-nofight.mp4"    
    vout="outputvideo.avi"
    cropDir=None #"output/cam_a"
    ## Create a detector
    configPath = "./yolo/yolov3.cfg"
    weightPath = "./yolo/yolov3.weights"
    metaPath = "./yolo/coco.data"
    classPath="./yolo/coco.names"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")
    if not os.path.exists(classPath):
        raise ValueError("Invalid className file path `" + os.path.abspath(classPath)+"`")

    #detector=DetectorYolo3_CPU(configPath,weightPath,classPath)
    detector=DetectorYolo3_GPU(configPath,weightPath,metaPath)

    ## Video streaming      
    if vin:
        cap = cv.VideoCapture(vin)
    else:
        cap = cv.VideoCapture(0)    
    window_name = "Detecting in progress"

    #cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty(window_name, cv.WND_PROP_AUTOSIZE, cv.WINDOW_AUTOSIZE)        
    cv.moveWindow(window_name,10,10)    
    frameW=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameH=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    print("Start detecting...")
    frameNo=0       
    fps=0.0

    writeVideo_flag = True 
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(vout, fourcc, 15, (frameW, frameH))

    while True:
        prev_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print('Cannot read frame from media')
            break       
        
        boxes=detector.predict(frame,'person',confThr,nmsThr,32,64)                 
        
        frame=DrawBoxes(boxes,frame)        
        label="frameNo: %08d" % (frameNo)
        cv.putText(frame,label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (128,0,255))        
        cv.imshow(window_name, frame) 
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)

        k=cv.waitKey(1)                
        if k==27:
            break
        if k==32:
            k=cv.waitKey()
        frameNo=frameNo+1   
        fps=(fps + 1/(time.time()-prev_time))/2
        print(fps)

    cap.release()
    if writeVideo_flag:
        out.release()

if __name__ == "__main__":
    Process()