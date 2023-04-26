# config for input and output
source_video_frame_path = '/nvme/liuwenran/datasets/playground/huangbo_fps10/'
middle_video_frame_path = 'results/controlnet_animation_frames/'
final_video_frame_path = 'results/final_frames/'
width = 512
height = 512

# config for controlnet animation
prompt = 'a man, best quality, extremely detailed'
negative_prompt = 'longbody, lowres, bad anatomy, ' + \
                'bad hands, missing fingers, extra digit, ' + \
                'fewer digits, cropped, worst quality, low quality'

# config for background generation
stable_diffusion_v15_url = 'Linaqruf/anything-v3.0'
model = dict(
    type='StableDiffusion',
    vae=dict(
        type='AutoencoderKL',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='vae'),
    unet=dict(
        type='UNet2DConditionModel',
        subfolder='unet',
        from_pretrained=stable_diffusion_v15_url),
    text_encoder=dict(
        type='ClipWrapper',
        clip_type='huggingface',
        pretrained_model_name_or_path=stable_diffusion_v15_url,
        subfolder='text_encoder'),
    tokenizer=stable_diffusion_v15_url,
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='EditDataPreprocessor'),
    init_cfg=dict(type='init_from_unet'),
    enable_xformers=False,
)

bg_seed = 24
bg_prompt = 'A party with crowded people, dancing'

# config for sam
point_coord = [[200, 160], [90, 370], [240, 370], [430, 370], [230, 450]]
sam_checkpoint = '/nvme/liuwenran/branches/liuwenran/dev-sdi/mmediting/resources/sam_model/sam_vit_h_4b8939.pth'
