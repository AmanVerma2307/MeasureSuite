import numpy as np

def read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the 
    # abundant bodys. 
    # read all lines into the pool to speed up, less io operation. 
    
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = [] 
    bodymat['njoints'] = njoints 

    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))

    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])    
        if bodycount == 0:
            continue 
        # skip the empty frame 
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)
            
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            
            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame,joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame,joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame,joint] = jointinfo[5:7]
    # prune the abundant bodys 
    for each in range(max_body):
        if each >= max(bodymat['nbodys']):
            if save_skelxyz:
                del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                del bodymat['depth_body{}'.format(each)]
    return bodymat 

def load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True 
    return missing_files 


def makeDict():

    """
    Function to create data dictionaries for both the sets

    OUTPUTS:-
    1) dataDict: Dictionary containing dictionary of action and identity values as keys and labels as values
    """

    action_ntu60 = {'A027':0,
                    'A007':1,
                    'A035':2,
                    'A028':3,
                    'A033':4,
                    'A049':5}
    
    action_ntu120 = {'A099':0,
                     'A098':1,
                     'A102':2,
                     'A069':3}

    id_ntu60 = {}
    id_ntu120 = {'P006':0,
                 'P008':1,
                 'P011':2}

    for id in range(1,41):
        if(id <= 9):
            idVal = 'P00'+str(id)
            id_ntu60.update({idVal:(id-1)})
        else:
            idVal = 'P0'+str(id)
            id_ntu60.update({idVal:(id-1)})

    for idx, id in enumerate(np.arange(41,107)):
        if(id <= 99):
            idVal = 'P0'+str(id)
            id_ntu120.update({idVal:idx+3})
        else:
            idVal = 'P'+str(id)
            id_ntu120.update({idVal:idx+3})

    return {'ntu_60':{'action':action_ntu60,
                      'id':id_ntu60},
            'ntu_120':{'action':action_ntu120,
                       'id':id_ntu120}
            }


def processSkeleton(filePath,
                    numFrames,
                    numJoints=25,
                    numSubjects=1):

    """
    Function to process skeleton

    INPUTS:-
    1) filePath: Path to input skeleton files
    2) numFrames: Number of frames in the input
    3) numJoints: Total number of joints
    4) numSubjects: Total number of subjects, Default: 1

    OUTPUTS:-
    1) sample: Processed output sample of shape - (3,numFrames,numJoints,numPerson)
    """

    sample = read_skeleton(filePath)['skel_body0']

    if(sample.shape[0] <= numFrames):
        sample = np.concatenate((sample,np.zeros((numFrames-sample.shape[0],numJoints,3))),axis=0)

    if(sample.shape[0] > numFrames):
        sample = sample[:numFrames]

    sample = np.expand_dims(sample,axis=-1)
    sample = np.transpose(sample,(2,0,1,3))
    return sample

if __name__ == "__main__":

    dataDict = makeDict()
    print(dataDict.items())
    print(dataDict['ntu_60']['action'],
          dataDict['ntu_60']['id'],
          dataDict['ntu_120']['action'],
          dataDict['ntu_120']['id'])

    # import matplotlib.pyplot as plt
    # import plotly.graph_objects as go

    # missingFile = load_missing_file('./utils/missingSkeletons.txt')
    # skateSamp = read_skeleton('./data/S001C001P001R001A001.skeleton')['skel_body0']
    sample = processSkeleton('./data/S001C001P001R001A027.skeleton',numFrames=120)
    print(sample.shape)

    # fig= go.Figure(go.Scatter3d(x=skateSamp[50,:,1],
    #                         y=skateSamp[50,:,2],
    #                         z=skateSamp[50,:,0], 
    #                         mode='lines', 
    #                         line_width=2, 
    #                         line_color='blue'))
    # fig.update_layout(width=600, height=600)
    # fig.show()

