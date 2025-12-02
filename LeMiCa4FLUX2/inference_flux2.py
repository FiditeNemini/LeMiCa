import torch,io
from diffusers import Flux2Pipeline
import time,os,re
import pandas as pd
import argparse

from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND, is_torch_npu_available, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import requests

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
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
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
    timestep = timestep.to(hidden_states.dtype) * 1000
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

    if is_torch_npu_available():
        freqs_cos_image, freqs_sin_image = self.pos_embed(img_ids.cpu())
        image_rotary_emb = (freqs_cos_image.npu(), freqs_sin_image.npu())
        freqs_cos_text, freqs_sin_text = self.pos_embed(txt_ids.cpu())
        text_rotary_emb = (freqs_cos_text.npu(), freqs_sin_text.npu())
    else:
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )
    
    # -------------------------lemica----------------------
    
    if self.enable_lemica:
        if hasattr(self, "bool_list") and len(self.bool_list) > self.cnt:
            should_calc = self.bool_list[self.cnt]
        else:
            should_calc = True

        self.cnt += 1
        self.store.append(should_calc)

        if self.cnt == self.num_steps:
            true_count = self.store.count(True)
            print(f'*** Total steps: {len(self.store)}, True count: {true_count}')
            self.store = []
            self.cnt = 0       
            
    if self.enable_lemica:
        
        
        if not should_calc:
            hidden_states += self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            # 4. Double Stream Transformer Blocks
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
            self.previous_residual = hidden_states - ori_hidden_states

    # 6. Output layers
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

def remote_text_encoder(prompts, device):
    """
    Calls the recently deployed FastAPI service and returns prompt_embeds (torch.Tensor).
    prompts: str or List[str]
    """
    TEXT_ENCODER_URL = "http://127.0.0.1:8006/predict"
    
    resp = requests.post(
        TEXT_ENCODER_URL,
        json={"prompt": prompts},
        timeout=600,
    )
    resp.raise_for_status()

    # Use torch.load for deserialization, same as in the official example
    prompt_embeds = torch.load(io.BytesIO(resp.content))

    # Move to the device used for current inference
    return prompt_embeds.to(device)
    
    
def get_args():
    parser = argparse.ArgumentParser(description="Run FLUX.2 with LeMiCa bool control.")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Enable caching: choose from [slow, medium, fast] or a numeric value. "
             "If omitted, caching is disabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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
            26: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 34, 41, 47, 49],
            20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 28, 35, 42, 49],
            15: [0, 1, 2, 3, 4, 5, 8, 9, 15, 21, 27, 33, 39, 45, 49],
        }

        if lemica_step not in calc_dict:
            raise ValueError(f"cache step {lemica_step} not in calc_dict")

        calc_list = calc_dict[lemica_step]
        bool_list = [i in calc_list for i in range(num_inference_steps)]
    else:
        lemica_step = None
        bool_list = None

    repo_id = "black-forest-labs/FLUX.2-dev"
    device = "cuda"
    torch_dtype = torch.bfloat16

    pipeline = Flux2Pipeline.from_pretrained(
        repo_id,
        text_encoder=None,
        torch_dtype=torch_dtype,
    )

    # === Caching control ===
    if args.cache is not None:
        print("[INFO] Cache is ENABLED.")
        print(f"[INFO] Using cache: {args.cache} -> {lemica_step}")
        Flux2Transformer2DModel.forward = Lemica_call
        pipeline.transformer.__class__.enable_lemica = True
        pipeline.transformer.__class__.cnt = 0
        pipeline.transformer.__class__.num_steps = num_inference_steps
        pipeline.transformer.__class__.bool_list = bool_list
        pipeline.transformer.__class__.previous_residual = None
        pipeline.transformer.__class__.store = []
    else:
        print("[INFO] Cache is DISABLED. Running pipeline without caching.")

    print("[INFO] Model loaded and ready.\n")
    pipeline.to(device)

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats()

    prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."


    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    image = pipeline(
        prompt_embeds=remote_text_encoder(prompt, device),
        generator=torch.Generator(device=device).manual_seed(seed),
        num_inference_steps=50,
        guidance_scale=4,
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

