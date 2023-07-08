# config for model
stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'

model = dict(
    type = 'ControlStableDiffusion',
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
    controlnet=dict(
        type='ControlNetModel',
        attention_head_dim = 8,
        block_out_channels = [320,640,1280,1280],
        conditioning_embedding_out_channels=[16,32,96,256],
        controlnet_conditioning_channel_order="rgb",
        cross_attention_dim = 768,
        down_block_types = ["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
        downsample_padding = 1,
        flip_sin_to_cos=True,
        freq_shift= 0,
        in_channels= 4,
        layers_per_block= 2,
        mid_block_scale_factor = 1,
        norm_eps= 1e-05,
        norm_num_groups= 32,
        only_cross_attention= False,
        resnet_time_scale_shift= "default",
        sample_size= 32,
        upcast_attention= False,
        use_linear_projection= False
        ),
    scheduler=dict(
        type='DDPMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    test_scheduler=dict(
        type='DDIMScheduler',
        from_pretrained=stable_diffusion_v15_url,
        subfolder='scheduler'),
    data_preprocessor=dict(type='DataPreprocessor'),
    init_cfg=dict(type='init_from_unet'))