from nuscenes.nuscenes import NuScenes

# Load data
nusc = NuScenes(version='v1.0-mini', dataroot='/ocean/projects/cis220039p/shared/nuscenes', verbose=True)

print("##### Data loaded ######")
# Scene MetaData
my_scene = nusc.scene[0]

# Take first sample from the scene 
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)


cam_f_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT_LEFT'])
print(cam_f_data['token'])
# print(f"My sample :{my_sample['data']['CAM_FRONT_LEFT']}")


# print(nusc.ego_pose[0])


# sensor = 'CAM_FRONT'
# cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
# cam_front_data