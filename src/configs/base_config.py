class BaseConfig:
    def __init__(self):
        
        ##############################
        ######## DATA CONFIGS ########
        ##############################

        self.tartanair_data_root = '/ocean/projects/cis220039p/shared/tartanair_v2'

        # Training data configs
        self.train_envs = [
            "ShoreCaves",
            "AbandonedFactory",
            "AbandonedSchool",
            "AmericanDiner",
            "AmusementPark",
            "AncientTowns",
            "Antiquity3D",
            "Apocalyptic",
            "ArchVizTinyHouseDay",
            "ArchVizTinyHouseNight",
            # "BrushifyMoon", # Seems to be very large and too easy for flow.
            "CarWelding",
            "CastleFortress",
            "ConstructionSite",
            "CountryHouse",
            "CyberPunkDowntown",
            "Cyberpunk",
            "DesertGasStation",
            "Downtown",
            "EndofTheWorld",
            "FactoryWeather",
            "Fantasy",
            "ForestEnv",
            "Gascola",
            "GothicIsland",
            # "GreatMarsh",
            "HQWesternSaloon",
            "HongKong",
            "Hospital",
            "House",
            "IndustrialHangar",
            "JapaneseAlley",
            "JapaneseCity",
            "MiddleEast",
            "ModUrbanCity",
            "ModernCityDowntown",
            "ModularNeighborhood",
            "ModularNeighborhoodIntExt",
            "NordicHarbor",
            "Ocean",
            "Office",
            "OldBrickHouseDay",
            "OldBrickHouseNight",
            "OldIndustrialCity",
            "OldScandinavia",
            "OldTownFall",
            "OldTownNight",
            "OldTownSummer",
            "OldTownWinter",
            # "PolarSciFi",
            "Prison",
            "Restaurant",
            "RetroOffice",
            "Rome",
            "Ruins",
            "SeasideTown",
            # "SeasonalForestAutumn",
            "SeasonalForestSpring",
            # "SeasonalForestSummerNight",
            "SeasonalForestWinter",
            # "SeasonalForestWinterNight",
            "Sewerage",
            "Slaughter",
            "SoulCity",
            "Supermarket",
            "TerrainBlending",
            "UrbanConstruction",
            "VictorianStreet",
            "WaterMillDay",
            "WaterMillNight",
            "WesternDesertTown",
            "AbandonedFactory2",
            "CoalMine"
        ]

        self.train_difficulties = []
        self.train_trajectory_ids = []

        # Val data configs
        self.val_envs = [
            "ShoreCaves",
            "MiddleEast",
            "AbandonedCable"
        ]

        self.val_difficulties = []
        self.val_trajectory_ids = []

        # Specify the modalities to load.
        self.modalities = ['image', 'pose', 'depth', 'flow']
        self.camnames = ['lcam_front']

        # Specify the dataloader parameters.
        self.new_image_shape_hw = None # If None, no resizing is performed. If a value is passed, then the image is resized to this shape.
        self.subset_framenum = 10 # This is the number of frames in a subset. Notice that this is an upper bound on the batch size. Ideally, make this number large to utilize your RAM efficiently. Information about the allocated memory will be provided in the console.
        self.seq_length = {'image': 2, 'pose': 2, 'depth': 2, 'flow': 1} # This is the length of the data-sequences. For example, if the sequence length is 2, then the dataloader will load pairs of images.
        self.seq_stride = 1 # This is the stride between the data-sequences. For example, if the sequence length is 2 and the stride is 1, then the dataloader will load pairs of images [0,1], [1,2], [2,3], etc. If the stride is 2, then the dataloader will load pairs of images [0,1], [2,3], [4,5], etc.
        self.frame_skip = 0 # This is the number of frames to skip between each frame. For example, if the frame skip is 2 and the sequence length is 3, then the dataloader will load frames [0, 3, 6], [1, 4, 7], [2, 5, 8], etc.
        self.batch_size = 4 # This is the number of data-sequences in a mini-batch.
        self.num_workers = 4 # This is the number of workers to use for loading the data.
        self.shuffle = True # Whether to shuffle the data. Let's set this to False for now, so that we can see the data loading in a nice video. Yes it is nice don't argue with me please. Just look at it! So nice. :)


        ##############################
        ####### MODEL CONFIGS ########
        ##############################
        self.load_pretrained_volnet = True
        self.volnet_checkpoint = "/ocean/projects/cis220039p/pkachana/projects/11-777-MultiModal-Machine-Learning-/vol/src/checkpoints/2024_12_12-18_45_43/model_step_71000.pt"

        self.load_pretrained_flow = False
        self.flow_checkpoint = "/ocean/projects/cis220039p/pkachana/projects/tartanvo-fisheye-old/src/tartanvo_fisheye/networks/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth" # GMFlow checkpoint

        # must be one of ['vit', 'resnet']
        self.pose_net_type = 'resnet'

        ##############################
        ######## TRAIN CONFIGS #######
        ##############################
        self.use_gt_flow = False
        self.freeze_flow_net = False
        self.supervise_flow = True

        assert not (self.use_gt_flow and self.supervise_flow), "Cannot use both gt flow and supervise flow"
        assert not (self.freeze_flow_net and self.supervise_flow), "Cannot freeze flow net and supervise flow"

        self.loss_alpha = 1.0

        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.num_steps = 100_000

        self.val_freq = 1000
        self.val_steps = 100


        ##############################
        ####### LOGGING CONFIGS ######
        ##############################
        self.ckpt_save_dir = "/ocean/projects/cis220039p/schawla1/VOLNet/src/checkpoints"
        self.project_name = 'VOL'
        self.log_freq = 10
        self.vis_freq = 100
        self.val_vis_freq = 10

class KittiConfig:
    def __init__(self):
        # self.vol_checkpoint = '/ocean/projects/cis220039p/schawla1/11-777-MultiModal-Machine-Learning-/vol/src/checkpoints/2024_12_12-19_27_43/model_step_0.pt'
        self.vol_checkpoint = '/ocean/projects/cis220039p/shared/vol/model_step_49000.pt'
        self.batch_size = 50    
        self.vol_input_shape = (3, 640, 640)
        # Logging Arguments
        self.project_name = 'VOL-KITTI-Evaluation'
        self.log_imgs = True

        # Data Directories
        self.calib_dir ='/ocean/projects/cis220039p/shared/datasets/KITTI_odometry/calib/dataset/sequences'
        self.image_dir ='/ocean/projects/cis220039p/shared/datasets/KITTI_odometry/color/dataset/sequences'
        self.lidar_dir ='/ocean/projects/cis220039p/shared/datasets/KITTI_odometry/velodyne/dataset/sequences'

        # KITTI Sequences to Evaluate
        self.sequences = [10]
