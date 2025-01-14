import math
import os
import os.path
import numpy as np
from PIL import Image

GT_POSE_PTH = '/ocean/projects/cis220039p/schawla1/VOLNet/src/kitti/ground_truth_pose/kitti_T_diff/'
class OdometryDataset():
    def __init__(self, calib_path = '/media/public_dataset/KITTI/odometry/data_odometry_calib/dataset/sequences', 
                 image_path = '/media/public_dataset/KITTI/odometry/data_odometry_color/dataset/sequences',
                 lidar_path = '/media/public_dataset/KITTI/odometry/data_odometry_velodyne/dataset/sequences', 
                 npoints=8192, is_training=True, random_seed = 3):
        
        self.random_seed = random_seed

        self.npoints = npoints
        self.is_training = is_training
        self.image_path = image_path
        self.lidar_path = lidar_path
        self.calib_path = calib_path

        # Fails with these vals 
        # self.len_list = [0, 4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000, 23201] 
        
        # Update length list to acccording to pcs/images in each of the sequence 
        self.len_list = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        self.file_map = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

        self.T_trans = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])



    def __getitem__(self, index):#

        if not self.is_training:
            np.random.seed(self.random_seed)

        for seq_idx, seq_num in enumerate(self.len_list):
            # TODO(@Shashwat): Hardcoded. Correct it
            if index < seq_num:
                cur_idx_pc2 = index
                if cur_idx_pc2 == 0:
                    cur_idx_pc1 = 0
                else:
                    cur_idx_pc1 = cur_idx_pc2 - 1        ###############    1 frame gap  ###############   
                
                cur_seq = seq_idx - 1
                if cur_seq < 0:
                    cur_seq = 0
                break        
        
                print(f"Index :{index}")
                cur_seq = seq_idx - 1
                if cur_seq < 0:
                    cur_seq = 0
                cur_idx_pc2 = self.len_list[seq_idx-1] - index 
                if cur_idx_pc2 == 0:
                    cur_idx_pc1 = 0
                else:
                    cur_idx_pc1 = cur_idx_pc2 - 1        ###############    1 frame gap  ###############   
                break        
                
        # TODO(@Shashwat): Make it work for multiple calib
        calib_path = os.path.join(self.calib_path, self.file_map[cur_seq], 'calib.txt')
        # print(f"calib_path :{calib_path}")
        Tr_path = calib_path
        Tr_data = self.read_calib_file(Tr_path)
        # TODO(@Shashwat): Parametrize
        # Note: Currently using img_2 which corresponds to left_img i.e P0. Change accordingly.
        P_left_cam = Tr_data['P0']
        P_left_cam = P_left_cam.reshape(3, 4)
        Tr_data = Tr_data['Tr']
        Tr = Tr_data.reshape(3,4)
        Tr = np.vstack((Tr, np.array([0, 0, 0, 1.0])))

        cur_lidar_dir = os.path.join(self.lidar_path, self.file_map[cur_seq])
        cur_image_dir = os.path.join(self.image_path, self.file_map[cur_seq])

        pc1_bin = os.path.join(cur_lidar_dir, 'velodyne/' + str(cur_idx_pc1).zfill(6) + '.bin')
        pc2_bin = os.path.join(cur_lidar_dir, 'velodyne/' + str(cur_idx_pc2).zfill(6) + '.bin')
        im1_png = os.path.join(cur_image_dir, 'image_2/' + str(cur_idx_pc1).zfill(6) + '.png')
        im2_png = os.path.join(cur_image_dir, 'image_2/' + str(cur_idx_pc2).zfill(6) + '.png')

        # print(im2_png)

        pose = np.load(GT_POSE_PTH + self.file_map[cur_seq] + '_diff.npy')

        point1 = np.fromfile(pc1_bin, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(pc2_bin, dtype=np.float32).reshape(-1, 4)

        img1 = Image.open(im1_png).convert('RGB')
        img2 = Image.open(im2_png).convert('RGB')
        img1 = np.array(img1).astype(np.float32) / 255. 
        img2 = np.array(img2).astype(np.float32) / 255. 

        if point1.shape[0] < point2.shape[0]:
            n = point1.shape[0]
        else:
            n = point2.shape[0] # num points of pc1 and pc2

        pos1 = point1[:n, :3]
        pos2 = point2[:n, :3]

        add = np.ones((n, 1))
        pos1 = np.concatenate([pos1, add], axis = -1)
        pos2 = np.concatenate([pos2, add], axis = -1)

        pos1 = np.matmul(Tr, pos1.T)
        pos2 = np.matmul(Tr, pos2.T)

        pos1 = pos1.T[ :, :3]
        pos2 = pos2.T[ :, :3]

        T_diff = pose[cur_idx_pc2 : cur_idx_pc2 + 1, :]

        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis = 0)   ##1*4
        T_diff = np.concatenate([T_diff, filler], axis=0)#  4*4


        is_ground = np.logical_or(pos1[:,1] > 1.1, pos1[:,1] > 1.1)

        not_ground = np.logical_not(is_ground)
        
        near_mask_x = np.logical_and(pos1[:, 0] < 30, pos1[:, 0] > -30)
        near_mask_y = np.logical_and(pos1[:, 2] < 30, pos1[:, 2] > -30) 
        near_mask = np.logical_and(near_mask_x, near_mask_y)

        near_mask = np.logical_and(not_ground, near_mask)
        indices_1 = np.where(near_mask)[0]
        

        is_ground = np.logical_or(pos2[:,1] > 1.1, pos2[:,1] > 1.1)

        not_ground = np.logical_not(is_ground)
        
        near_mask_x = np.logical_and(pos2[:, 0] < 30, pos2[:, 0] > -30)
        near_mask_y = np.logical_and(pos2[:, 2] < 30, pos2[:, 2] > -30)            
        near_mask = np.logical_and(near_mask_x, near_mask_y)

        near_mask = np.logical_and(not_ground, near_mask)
        indices_2 = np.where(near_mask)[0]


        if len(indices_1) >= self.npoints:
            sample_idx1 = np.random.choice(indices_1, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((indices_1, np.random.choice(indices_1, self.npoints - len(indices_1), replace=True)), axis=-1)
    

        if len(indices_2) >= self.npoints:
            sample_idx2 = np.random.choice(indices_2, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((indices_2, np.random.choice(indices_2, self.npoints - len(indices_2), replace=True)), axis=-1)
    

        pos1 = pos1[sample_idx1, :]
        pos2 = pos2[sample_idx2, :]
        
        n = pos1.shape[0]

        add_T = np.ones((n, 1))
        pos2_gt = np.concatenate([pos2, add_T], axis = -1)
        pos2_gt = np.matmul(T_diff, pos2_gt.T)
        pos2_gt = pos2_gt.T[ :, :3]
        T_gt = T_diff

        
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3:]

        z_gt, y_gt, x_gt = self.mat2euler( M = R_gt)
        q_gt = self.euler2quat(z = z_gt, y = y_gt, x = x_gt)

        # pos2: (h, w, 3)
        # pos1: (h, w, 3)
        # pos2: (npoints, 3)
        # pos1: (npoints, 3)

        return img2, img1, pos2, pos1, q_gt, t_gt, P_left_cam

    def __len__(self):
        return sum(self.len_list)

    def feed_random(self, random_seed):
        self.random_seed = random_seed
        return 0
    
    def read_calib_file(self,path):  # changed

        float_chars = set("0123456789.e+- ")
        data = {}

        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass
        return data
    
    
    
    def euler2mat(self, anglex, angley, anglez):

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        
        R_trans = Rx.dot(Ry).dot(Rz)

        return R_trans


    def mat2euler(self, M, cy_thresh=None, seq='zyx'):

        M = np.asarray(M)
        if cy_thresh is None:
            cy_thresh = np.finfo(M.dtype).eps * 4

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33*r33 + r23*r23)
        if seq=='zyx':
            if cy > cy_thresh: # cos(y) not close to zero, standard form
                z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
                y = math.atan2(r13,  cy) # atan2(sin(y), cy)
                x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
            else: # cos(y) (close to) zero, so x -> 0.0 (see above)
                # so r21 -> sin(z), r22 -> cos(z) and
                z = math.atan2(r21,  r22)
                y = math.atan2(r13,  cy) # atan2(sin(y), cy)
                x = 0.0
        elif seq=='xyz':
            if cy > cy_thresh:
                y = math.atan2(-r31, cy)
                x = math.atan2(r32, r33)
                z = math.atan2(r21, r11)
            else:
                z = 0.0
                if r31 < 0:
                    y = np.pi/2
                    x = math.atan2(r12, r13)
                else:
                    y = -np.pi/2
        else:
            raise Exception('Sequence not recognized')
        return z, y, x

    def euler2quat(self, z=0, y=0, x=0, isRadian=True):
        ''' Return quaternion corresponding to these Euler angles
        Uses the z, then y, then x convention above
        Parameters
        ----------
        z : scalar
            Rotation angle in radians around z-axis (performed first)
        y : scalar
            Rotation angle in radians around y-axis
        x : scalar
            Rotation angle in radians around x-axis (performed last)
        Returns
        -------
        quat : array shape (4,)
            Quaternion in w, x, y z (real, then vector) format
        Notes
        -----
        We can derive this formula in Sympy using:
        1. Formula giving quaternion corresponding to rotation of theta radians
            about arbitrary axis:
            http://mathworld.wolfram.com/EulerParameters.html
        2. Generated formulae from 1.) for quaternions corresponding to
            theta radians rotations about ``x, y, z`` axes
        3. Apply quaternion multiplication formula -
            http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
            formulae from 2.) to give formula for combined rotations.
        '''
    
        if not isRadian:
            z = ((np.pi)/180.) * z
            y = ((np.pi)/180.) * y
            x = ((np.pi)/180.) * x
        z = z/2.0
        y = y/2.0
        x = x/2.0
        cz = math.cos(z)
        sz = math.sin(z)
        cy = math.cos(y)
        sy = math.sin(y)
        cx = math.cos(x)
        sx = math.sin(x)
        return np.array([
                        cx*cy*cz - sx*sy*sz,
                        cx*sy*sz + cy*cz*sx,
                        cx*cz*sy - sx*cy*sz,
                        cx*cy*sz + sx*cz*sy])
