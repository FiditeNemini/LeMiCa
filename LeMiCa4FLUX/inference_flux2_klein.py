import torch,io
import time,os,re
import pandas as pd
import argparse

from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND, is_torch_npu_available, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import requests
from diffusers import Flux2KleinPipeline



def Lemica_call(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:

    # 0. Handle input arguments
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    num_txt_tokens = encoder_hidden_states.shape[1]

    # 1. Calculate timestep embedding and modulation parameters
    # print(timestep)
    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)[0]

    # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 3. Calculate RoPE embeddings from image and text tokens
    # NOTE: the below logic means that we can't support batched inference with images of different resolutions or
    # text prompts of differents lengths. Is this a use case we want to support?
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    if self.enable_lemica:    
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
            
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        double_stream_mod_img,
                        double_stream_mod_txt,
                        concat_rotary_emb,
                        joint_attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb_mod_params_img=double_stream_mod_img,
                        temb_mod_params_txt=double_stream_mod_txt,
                        image_rotary_emb=concat_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
            # Concatenate text and image streams for single-block inference
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            # 5. Single Stream Transformer Blocks
            for index_block, block in enumerate(self.single_transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        None,
                        single_stream_mod,
                        concat_rotary_emb,
                        joint_attention_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=None,
                        temb_mod_params=single_stream_mod,
                        image_rotary_emb=concat_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
            # Remove text tokens from concatenated stream
            hidden_states = hidden_states[:, num_txt_tokens:, ...]
            
            # Store residuals for future use in CFG-aware cache state
            cache_state['previous_encoder_residual'] = (encoder_hidden_states.to(cache_device) - ori_encoder_hidden_states)
            cache_state['previous_hidden_residual'] = (hidden_states.to(cache_device) - ori_hidden_states)
            pass  # Residuals calculated and stored            

    # 6. Output layers
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)



def get_args():
    parser = argparse.ArgumentParser(description="Run FLUX.2 with LeMiCa bool control.")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Enable caching: choose from [slow, medium, fast] or a numeric value. "
             "If omitted, caching is disabled.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    # === Parse args ===
    args = get_args()
    num_inference_steps = 50
    seed = args.seed

    # === Speed modes ===
    speed_modes = {
        "slow": 26,
        "medium": 20,
        "fast": 15,
        "ultra": 10,
    }

    # === Resolve cache setting only if --cache is provided ===
    if args.cache is not None:
        cache_key = args.cache.lower()
        if cache_key in speed_modes:
            lemica_step = speed_modes[cache_key]
        elif cache_key.isdigit():
            lemica_step = int(cache_key)
        else:
            raise ValueError(
                f"Invalid cache value: {args.cache}. Must be one of "
                f"{list(speed_modes.keys())} or a number."
            )
        
        calc_dict = {
            26: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 35, 43, 47, 49],
            20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 32, 42, 47, 49],
            15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 25, 33, 41, 49],
            10: [0, 1, 4, 5, 9, 17, 25, 33, 41, 49],
        }

        if lemica_step not in calc_dict:
            raise ValueError(f"cache step {lemica_step} not in calc_dict")

        calc_list = calc_dict[lemica_step]
        bool_list = [i in calc_list for i in range(num_inference_steps)]
    else:
        lemica_step = None
        bool_list = None

        
    device = "cuda"        
    dtype = torch.bfloat16
    pipeline = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=dtype)

    pipeline.to(device)

    # === Caching control ===
    if args.cache is not None:
        print("[INFO] Cache is ENABLED.")
        print(f"[INFO] Using cache: {args.cache} -> {lemica_step}")
        Flux2Transformer2DModel.forward = Lemica_call
        pipeline.transformer.__class__.enable_lemica = True
        pipeline.transformer.__class__.cnt = 0
        pipeline.transformer.__class__.num_steps = num_inference_steps
        pipeline.transformer.__class__.should_calc_list = bool_list
        pipeline.transformer.__class__.pair_cnt = 0        

    else:
        print("[INFO] Cache is DISABLED. Running pipeline without caching.")

    print("[INFO] Model loaded and ready.\n")
    # pipeline.to(device)
    # pipeline.enable_model_cpu_offload()    
    

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats()

    prompt =  "A cat holding a sign that says hello world"
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    image = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=50,
        generator=torch.Generator(device=device).manual_seed(seed)
    ).images[0]

    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) * 1e-3
    peak_memory = torch.cuda.max_memory_allocated(device=device)

    if args.cache is not None:
        image.save(f"flux2_output_{lemica_step}.png")
    else:
        image.save("flux2_output.png")

    print(
        f"epoch time: {elapsed_time:.2f} sec, "
        f"parameter memory: {parameter_peak_memory/1e9:.2f} GB, "
        f"memory: {peak_memory/1e9:.2f} GB"
    )

if __name__ == "__main__":
    main()

