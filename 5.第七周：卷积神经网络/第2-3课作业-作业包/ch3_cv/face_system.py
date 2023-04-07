#from curses.textpad import rectangle
from lib2to3.pgen2.token import NAME
from operator import truediv
import os
import json
from pickle import FALSE, NONE, TRUE
from pyexpat import features
from queue import Empty
from turtle import update
from unittest import result
import cv2
import numpy as np
import traceback

from PIL import Image
from PIL import ImageDraw, ImageFont
import onnxruntime
from face_detector import Detector
from face_landmark import LandmarksExtractor
from arcface import ArcFace,sub_feature,l2_norm, face_distance


class Face_data(object):
    _DATA_INFORMATION={}
    def __init__(self,path_json=None,db_dict=None)->None:
       if not path_json is None:
            self.load_json(path_json)
       if not db_dict is None:
            self.update(db_dict)
    def query_NtoN(self,features_to_check,features_known=None,threshold=0.6):
        features_to_check=np.ascontiguousarray(features_to_check)
        features_known=(self.known_mean_features() if features_known is None else features_known)
        #print(features_to_check)
        #print(features_known)
        #print(self._DATA_INFORMATION)
        dists_N=np.dot(features_to_check,features_known)
        #print(dists_N)
        dists_max=dists_N.max(axis=-1)#weidu -1
        inds=dists_N.argmax(axis=-1)
        knowns=dists_max>threshold

        #print(dists_max)
        return inds,knowns,dists_max
    def query(self,feature_to_check,known_features=None,threshold=0.6):
        inds,knowns,dists_max=self.query_NtoN([feature_to_check],known_features,threshold)
        return inds[0],knowns[0],dists_max[0]
    def data_dict(self):
        return self._DATA_INFORMATION.copy()
    def known_mean_features(self):
        list_features=[d["feature_vector"] for d in self._DATA_INFORMATION.values()]
        return np.ascontiguousarray(list_features).T

    def index2id(self):
        return {ind: id_person for ind, id_person in enumerate(self._DATA_INFORMATION.keys())}

    def id2name(self):
        return {k: v["name"] for k, v in self._DATA_INFORMATION.items()}

    def ind2name(self):
        return {ind: v["name"] for ind, v in enumerate(self._DATA_INFORMATION.values())}

    def all_names(self):
        return [v["name"] for v in self._DATA_INFORMATION.values()]

    def nb_people(self):
        return len(self._DATA_INFORMATION.keys())

    def append(self, id, info_dict):
        self._DATA_INFORMATION.update({id: info_dict})
    def update(self, db_dict):
        self._DATA_INFORMATION.update(db_dict)

    def load_json(self, path_json):
        with open(path_json, "r") as fp:
            db_dict = json.load(fp)
        self.update(db_dict)
    def save_to_json(self, path_json):
        with open(path_json, "w") as fp:
            json.dump(self.db_dict)
class Face(object):
    _INFORMATION={"name": None, "id": None,
                  "gender_id": None, "feature_vector": None, 'feature_list': None}
    def update(self,info_dict):
        self._INFORMATION.update(info_dict)
    def save(self,path_json):
        try:
            with open(path_json,'w') as fp:
                json.dump(self._INFORMATION,fp)
        except:
            traceback.print_exc()
            os.remove(path_json)
    def load(self,info_path):
        with open(info_path,'r')as fp:
            info_dic=json.load(fp)
        for key in info_dic.keys():
            if not key in self._INFORMATION.keys():
                raise KeyError("Info key:{} is not aviliable in Face".format(key))
        self._INFORMATION.update(info_dic)
        return self._INFORMATION.copy()
def parse_filename(path_dir):#返回名字(字符串),ID(数字),性别(男0女1未知2)
    name_dir=os.path.split(path_dir)[-1]
    Name,Id,Gender_char=name_dir.split('_')
    Gender={'m':0,'f':1,'u':2}[Gender_char]
    Id=int(Id)
    return Name,Id,Gender
def Read(filename):
    return cv2.imdecode(np.fromfile(file=filename,dtype=np.uint8),cv2.IMREAD_COLOR)
def remove_old(root_dir):
    pathes_dir=[path_dir for path_dir in [os.path.join(root_dir,name_dir)for name_dir in os.listdir(root_dir)]if os.path.isdir(path_dir)]
    for path_dir in pathes_dir:
        info_path=os.path.join(path_dir,'infomation.json')
        if os.path.exists(info_path):
            os.remove(info_path)

def work_dir(path_dir,detector:Detector,lm_extractor:LandmarksExtractor,arcface:ArcFace,over_write=False):
    face=Face()
    print('Now begin regist dir:{}'.format(path_dir))
    info_path=os.path.join(path_dir,'infomation.json')#信息存储路径
    empty_path_flag=os.path.join(path_dir,'empty.flag')
    imgs_num=0
    Name,Id,Gender=parse_filename(path_dir)
    
    if((not os.path.exists(info_path)) or over_write):
        if os.path.exists(info_path):
            os.remove(info_path)
        path_list=[os.path.join(path_dir,fn)for fn in os.listdir(path_dir)]
        feature_list=[]
        for path in path_list:#提取每张图片信息
            try:
                if path[-4:] in ['flag','json']:
                    continue
                img_src = Read(path)#获取图片
                if img_src is None:
                    continue
                rectangles,tmp=detector.predict(img_src)#寻找人脸
                if len(rectangles)==1:
                    lm=lm_extractor.predict(img_src,rectangles)[0]
                    feature_now=arcface.predict(img_src,[lm])[0]
                    feature_list.append(feature_now)
                    imgs_num+=1
                else:
                    continue
            except KeyboardInterrupt:
                traceback.print_exc()
                quit()
            except:
                traceback.print_exc()
                continue
        if len(feature_list):
            feature_list=np.asarray(feature_list)
            feature_list,mean_feature=sub_feature(feature_list)
            
            print("Name: {}    Id: {}    Gender: {}    \nFacial feature get from {} images from directory {} with {} faces".format(Name,
                Id,Gender,imgs_num,path_dir,len(feature_list)))

            info_dict = {"name": Name, "id": Id, "gender_id": Gender,
                         "feature_vector": mean_feature.tolist(), "feature_list": feature_list.tolist()}
            face.update(info_dict)
            face.save(path_json=info_path)

            flag_empty=False
            if(os.path.exists(empty_path_flag)):
                os.remove(empty_path_flag)
        else:
            flag_empty=True
            with open(empty_path_flag,'w')as fp:
                fp.write('e')
            mean_feature=None
    elif os.path.exists(empty_path_flag):
        flag_empty=True
        mean_feature=None
    else:
        print('Loading info from {}'.format(info_path))
        info_dict=face.load(info_path)
        flag_empty=False
        if os.path.exists(empty_path_flag):
            os.remove(empty_path_flag)
    return face._INFORMATION,flag_empty

def register_Root_dir(root_dir,out_path,detector:Detector,lm_extractor:LandmarksExtractor,arcface:ArcFace,over_write=False):
    if over_write:
        remove_old(root_dir)
    pathes_dir=[path_dir for path_dir in [os.path.join(root_dir,name_dir)for name_dir in os.listdir(root_dir)]if os.path.isdir(path_dir)]
    results=[]
    print("Now begin batch register {}".format(root_dir))
    for path_dir in pathes_dir:
        A,B=work_dir(path_dir,detector,lm_extractor,arcface,over_write)
        results.append([A.copy(),B])
    db_dict={}
    for (information,empty_flag) in results:
        if not empty_flag:
            db_dict.setdefault(information["id"],information)
    with open(out_path,"w")as fp:
        json.dump(db_dict,fp)
    print("Now batch register is over")
    return db_dict

def detector_face_from_image(img_path,detector:Detector,lm_extractor:LandmarksExtractor,arcface:ArcFace,face_data:Face_data):
    #get image
    img_to_detect=cv2.imread(img_path)
    size=img_to_detect.shape
    a=size[1]
    b=size[0]#1000,500
    s1=1500./int(a)
    s2=600./int(b)
    if(s2<s1):s1=s2
    img_to_detect=cv2.resize(img_to_detect,None,fx=s1,fy=s1)
    boxes,confidences=detector.predict(img_to_detect)
    landmarks=lm_extractor.predict(img_to_detect,boxes)
    Result=arcface.predict(img_to_detect,landmarks)
    inds, knowns, dists_max = face_data.query_NtoN(Result)
    img_to_show=img_to_detect
    image=Image.fromarray(cv2.cvtColor(img_to_show,cv2.COLOR_BGR2RGB))
    fontText=ImageFont.truetype("data/fonts/SourceHanSansHWSC-Regular.otf", 18, encoding="utf-8")
    for rect,ind,known,score,landmarks in zip(boxes,inds,knowns,dists_max,landmarks):
        rect=list(map(int ,rect))
        if(score>0.7):
            cv2.rectangle(img_to_detect,(rect[0],rect[1]),(rect[2],rect[3]),(0,0,255),2)
            cx=rect[0]
            cy=rect[1]+12
            name =face_data.ind2name()[ind]
            image=Image.fromarray(cv2.cvtColor(img_to_show,cv2.COLOR_BGR2RGB))
            draw= ImageDraw.Draw(image)    
            draw.text((cx,cy+32),name,font=fontText,fill=(0,255,0))
            img_to_show=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        for(x,y)in landmarks:
            cv2.circle(img_to_show,(int(x),int(y)),1,(255,255,0),-1)
        #print(score)
    
    cv2.imshow("img",img_to_show)
    cv2.waitKey()
def detector_root_dir(root_dir,detector:Detector,lm_extractor:LandmarksExtractor,arcface:ArcFace,face_data:Face_data):
    print("Now begin:")
    pathes_dir=[os.path.join(root_dir,name_dir)for name_dir in os.listdir(root_dir)]
    print(pathes_dir)
    for path_dir in pathes_dir:
        print("NOW")
        print(path_dir)
        detector_face_from_image(path_dir,detector,lm_extractor,arcface,face_data)

if __name__=="__main__":
    Now=onnxruntime.SessionOptions()
    Now.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    detector=Detector("weights/face_detector_640_dy_sim.onnx",input_size=(640,480),top_k=16)
    lm_extractor = LandmarksExtractor("weights/landmarks_68_pfld_dy_sim.onnx")
    arcface = ArcFace("weights/arc_mbv2_ccrop_sim.onnx")

    root_dir = "data/imgs_celebrity"
    out_path = "t.json"
    db_dict=register_Root_dir(root_dir,out_path, detector, lm_extractor,
                 arcface, over_write=True)
    #print(db_dict)
    
    face_data = Face_data(db_dict=db_dict)# 实例化FacialDB人脸数据库类
    all_path="data/detect"
    detector_root_dir(all_path,detector,lm_extractor,arcface,face_data)

    