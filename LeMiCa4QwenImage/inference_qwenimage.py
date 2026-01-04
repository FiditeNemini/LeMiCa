from typing import Optional, List, Dict, Any, Tuple, Union
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.models import QwenImageTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
import torch
import time,os,re
import pandas as pd
import argparse


def LeMiCa_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,  # TODO: this should probably be removed
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:

    
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.img_in(hidden_states)

    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
    
    cache_device = hidden_states.device
            
    
    is_positive_prompt = (self.pair_cnt % 2 == 0)
    cache_key = 'positive' if is_positive_prompt else 'negative'
    self.pair_cnt += 1
    

    if not hasattr(self, 'lexcache_states'):
        self.lexcache_states = {
            'positive': {'accumulated_rel_l1_distance': 0, 'previous_encoder_residual': None, 'previous_hidden_residual': None},
            'negative': {'accumulated_rel_l1_distance': 0, 'previous_encoder_residual': None, 'previous_hidden_residual': None}
        }        
            
    cache_state = self.lexcache_states[cache_key]
    
    # LeMiCa
    should_calc = self.should_calc_list[self.cnt]
    if cache_key == 'negative':
        self.cnt += 1 
        if self.cnt == self.num_steps:
            self.cnt = 0
            self.pair_cnt = 0
    
    if not self.enable_cache:
        should_calc = True    
        
    # print('***',cache_key, self.cnt, should_calc) 

    if not should_calc:
        # Use CFG-aware cached residuals
        if (cache_state['previous_encoder_residual'] is not None and 
            cache_state['previous_hidden_residual'] is not None):
            # Check if cached residuals have compatible shapes
            if (cache_state['previous_encoder_residual'].shape == encoder_hidden_states.shape and 
                cache_state['previous_hidden_residual'].shape == hidden_states.shape):
                pass  # Using cached computation
                encoder_hidden_states += cache_state['previous_encoder_residual'].to(encoder_hidden_states.device)
                hidden_states += cache_state['previous_hidden_residual'].to(hidden_states.device)
            else:
                pass  # Shape mismatch, forcing recalculation
                should_calc = True
        else:
            pass  # No cached residuals available
            should_calc = True


    if should_calc:

        ori_encoder_hidden_states = encoder_hidden_states.to(cache_device)
        ori_hidden_states = hidden_states.to(cache_device)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # Store residuals for future use in CFG-aware cache state
        cache_state['previous_encoder_residual'] = (encoder_hidden_states.to(cache_device) - ori_encoder_hidden_states)
        cache_state['previous_hidden_residual'] = (hidden_states.to(cache_device) - ori_hidden_states)
        pass  # Residuals calculated and stored

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
        
    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


class QwenImageModel():
    """
    A text-to-image model implementation using Qwen's diffusion pipeline.
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen-Image", device: Optional[str] = None) -> None:
        """
        Initialize the Qwen image model.
        
        Args:
            model_path (str): The path to the model (default is "Qwen/Qwen-Image")
            device (Optional[str]): The device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Determine device and dtype
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Load the pipeline
        self.pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=self.torch_dtype)
        self.pipe = self.pipe.to(self.device)
        # self.model_path = model_path
        
        # Positive magic prompts for different languages
        self.positive_magic = {
            "en": "Ultra HD, 4K, cinematic composition.",
            "zh": "超清，4K，电影级构图"
        }
        
        # ---------------------------------------- CACHE BELOW ------------------------------------------- 
        
        if 'Qwen-Image-2512' in model_path:
            speed_dict = {
                "slow":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 23, 28, 34, 39, 42, 44, 46, 47, 48, 49],
                "medium": [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 19, 26, 33, 40, 46, 48, 49],
                "fast":   [0, 2, 4, 8, 15, 22, 29, 36, 43, 49]
            }
        else:
            speed_dict = {
                "slow":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 33, 36, 42, 49],
                "medium": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 22, 29, 36, 42, 49],
                "fast":   [0, 1, 3, 7, 14, 21, 28, 35, 42, 49]
            }

        
        num_inference_steps = 50
        bool_lists = {
            speed: [(i in steps) for i in range(num_inference_steps)]
            for speed, steps in speed_dict.items()
        }
        self.bool_lists = bool_lists
        self.pipe.transformer.__class__.enable_cache = True
        self.pipe.transformer.__class__.cnt = 0
        self.pipe.transformer.__class__.num_steps = num_inference_steps
        self.pipe.transformer.__class__.should_calc_list = []
        self.pipe.transformer.__class__.pair_cnt = 0
        # ---------------------------------------- CACHE ABOVE -----------------------------------------           

    
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = '',
        height: int = 928,
        width: int = 1664,
        num_images_per_prompt: int = 1,
        cache: Optional[str] = None,
        seed: int = 42,
        **kwargs: Dict[str, Any]
    ) -> List[Image.Image]:
        """Generate images based on the given prompt."""
        
        lang = "zh" if any('\u4e00' <= char <= '\u9fff' for char in prompt) else "en"
        full_prompt = prompt + " " + self.positive_magic[lang]
        
        # Default params
        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = 50
        if "true_cfg_scale" not in kwargs:
            kwargs["true_cfg_scale"] = 4.0

        # Always set generator with given seed
        kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)
        print(f"Seed: {seed}")

        # Cache control
        if cache:
            QwenImageTransformer2DModel.forward = LeMiCa_forward
            self.pipe.transformer.__class__.should_calc_list = self.bool_lists[cache]
            steps_enabled = sum(self.bool_lists[cache])
            total_steps = len(self.bool_lists[cache])
            print(f"[Cache Enabled] Mode: {cache}")
        else:
            print(f"[Cache Disabled] No cache mode active")

        # Generate
        start_time = time.time()
        output = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs
        )
        end_time = time.time()
        print(f"Latency: {end_time - start_time:.3f} s")
        return output.images
    
def get_args():
    parser = argparse.ArgumentParser(description="Run Qwen-Image batch generation with cache acceleration.")
    parser.add_argument(
        "--cache", type=str, default=None,
        help="Acceleration mode: choose from ['slow', 'medium', 'fast'] or None for no cache.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()
    

if __name__=="__main__":
    args = get_args()

    # model_name = "Qwen/Qwen-Image"  
    model_name = "Qwen/Qwen-Image-2512"

    model = QwenImageModel(model_path = model_name)
    # prompt = "一位穿着印有“LeMiCa”标志黑色T恤的中国女性，手持黑色马克笔，微笑着正对镜头。她身后是数据中心机房的玻璃墙，墙上手写体清晰地写着:'LeMiCa 注意到局部误差累积问题，创新性提出利用 DAG 和 路径优化 以提升加速后的生成一致性' 这些内容。"
    
    prompt = '''A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.'''
    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    images = model(prompt, negative_prompt = negative_prompt, num_images_per_prompt=1, cache=args.cache, seed=args.seed)
    cache_name = args.cache if args.cache else "noCache"
    for i, img in enumerate(images):
        img.save(f"image_{cache_name}_{i}.png")
    print(f"Generated {len(images)} images with cache mode '{cache_name}'.")

    
    

    

    

    

