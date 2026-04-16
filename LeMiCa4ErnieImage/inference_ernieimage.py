import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import ErnieImagePipeline
from diffusers.models import ErnieImageTransformer2DModel


try:
    from diffusers.models.transformers.transformer_ernie_image import ErnieImageTransformer2DModelOutput
except Exception:
    @dataclass
    class ErnieImageTransformer2DModelOutput:
        sample: torch.Tensor


def get_qwen_style_calc_dict() -> Dict[int, List[int]]:
    return {
        25: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 31, 38, 43, 46, 47, 48, 49],
        17: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 27, 37, 44, 48, 49],
        10: [0, 1, 3, 7, 14, 21, 28, 35, 42, 49],
    }


def resolve_cache_plan(cache: str, num_inference_steps: int) -> Tuple[str, List[bool], int]:
    cache = cache.lower()
    speed_modes = {"slow": 25, "medium": 17, "fast": 10}
    calc_dict = get_qwen_style_calc_dict()

    if cache in speed_modes:
        target_step = speed_modes[cache]
    elif cache.isdigit():
        target_step = int(cache)
    else:
        raise ValueError(
            f"Invalid --cache value: {cache}. Use one of {list(speed_modes.keys())} or one of {list(calc_dict.keys())}."
        )

    if target_step not in calc_dict:
        raise ValueError(f"cache step {target_step} not found in calc_dict: {sorted(calc_dict.keys())}")

    calc_steps = calc_dict[target_step]
    bool_list = [i in calc_steps for i in range(num_inference_steps)]
    if num_inference_steps > 0:
        bool_list[0] = True
        bool_list[-1] = True
    return cache, bool_list, target_step


def lemica_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    text_bth: torch.Tensor,
    text_lens: torch.Tensor,
    return_dict: bool = True,
):
    device, dtype = hidden_states.device, hidden_states.dtype
    batch_size, _, height, width = hidden_states.shape
    patch_size = self.patch_size
    patch_h, patch_w = height // patch_size, width // patch_size
    image_token_count = patch_h * patch_w

    image_tokens = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
    if self.text_proj is not None and text_bth.numel() > 0:
        text_bth = self.text_proj(text_bth)
    text_token_count = text_bth.shape[1]
    text_tokens = text_bth.transpose(0, 1).contiguous()

    merged_tokens = torch.cat([image_tokens, text_tokens], dim=0)
    seq_len = merged_tokens.shape[0]

    text_ids = (
        torch.cat(
            [
                torch.arange(text_token_count, device=device, dtype=torch.float32)
                .view(1, text_token_count, 1)
                .expand(batch_size, -1, -1),
                torch.zeros((batch_size, text_token_count, 2), device=device),
            ],
            dim=-1,
        )
        if text_token_count > 0
        else torch.zeros((batch_size, 0, 3), device=device)
    )
    grid_yx = torch.stack(
        torch.meshgrid(
            torch.arange(patch_h, device=device, dtype=torch.float32),
            torch.arange(patch_w, device=device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    image_ids = torch.cat(
        [
            text_lens.float().view(batch_size, 1, 1).expand(-1, image_token_count, -1),
            grid_yx.view(1, image_token_count, 2).expand(batch_size, -1, -1),
        ],
        dim=-1,
    )
    rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

    valid_text = (
        torch.arange(text_token_count, device=device).view(1, text_token_count) < text_lens.view(batch_size, 1)
        if text_token_count > 0
        else torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
    )
    attention_mask = torch.cat(
        [torch.ones((batch_size, image_token_count), device=device, dtype=torch.bool), valid_text], dim=1
    )[:, None, None, :]

    timestep_emb = self.time_proj(timestep).to(dtype=dtype)
    cond = self.time_embedding(timestep_emb)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
        x.unsqueeze(0).expand(seq_len, -1, -1).contiguous() for x in self.adaLN_modulation(cond).chunk(6, dim=-1)
    ]
    temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]

    enable_lemica = bool(getattr(self, "enable_lemica", False))
    should_calc = True

    if enable_lemica:
        if not hasattr(self, "store"):
            self.store = []
        cnt = int(getattr(self, "cnt", 0))
        bool_list = getattr(self, "bool_list", None)
        if bool_list is not None and cnt < len(bool_list):
            should_calc = bool_list[cnt]

        self.store.append(bool(should_calc))
        self.cnt = cnt + 1

        num_steps = int(getattr(self, "num_steps", 0))
        if num_steps > 0 and self.cnt >= num_steps:
            if bool(getattr(self, "verbose", True)):
                true_count = sum(self.store)
                print(f"[LeMiCa] total steps: {len(self.store)}, computed: {true_count}")
            self.store = []
            self.cnt = 0

        if not should_calc:
            prev_residual = getattr(self, "previous_residual", None)
            if prev_residual is not None and prev_residual.shape == merged_tokens.shape:
                merged_tokens = merged_tokens + prev_residual.to(merged_tokens.device, merged_tokens.dtype)
            else:
                should_calc = True

    if not enable_lemica or should_calc:
        origin_tokens = merged_tokens.clone() if enable_lemica else None
        for layer in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                merged_tokens = self._gradient_checkpointing_func(layer, merged_tokens, rotary_pos_emb, temb, attention_mask)
            else:
                merged_tokens = layer(merged_tokens, rotary_pos_emb, temb, attention_mask)

        if enable_lemica:
            self.previous_residual = (merged_tokens - origin_tokens).detach()

    merged_tokens = self.final_norm(merged_tokens, cond).type_as(merged_tokens)
    patches = self.final_linear(merged_tokens)[:image_token_count].transpose(0, 1).contiguous()
    output = (
        patches.view(batch_size, patch_h, patch_w, patch_size, patch_size, self.out_channels)
        .permute(0, 5, 1, 3, 2, 4)
        .contiguous()
        .view(batch_size, self.out_channels, height, width)
    )

    if return_dict:
        return ErnieImageTransformer2DModelOutput(sample=output)
    return (output,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeMiCa acceleration for ErnieImage (transformer-only patch).")
    parser.add_argument("--model-path", type=str, default="baidu/ERNIE-Image", help="ErnieImage model path.")
    parser.add_argument("--cache", type=str, default=None, help="slow | medium | fast | 25 | 17 | 10")
    parser.add_argument("--prompt", type=str, default="A cute panda reading a book in a bamboo forest.")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-pe", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs_ernie_lemica")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu. Defaults to auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = ErnieImagePipeline.from_pretrained(args.model_path, torch_dtype=dtype).to(device)

    if args.cache is not None:
        cache_name, bool_list, target_step = resolve_cache_plan(args.cache, args.num_inference_steps)
        ErnieImageTransformer2DModel.forward = lemica_forward
        transformer_cls = pipe.transformer.__class__
        transformer_cls.enable_lemica = True
        transformer_cls.cnt = 0
        transformer_cls.num_steps = args.num_inference_steps
        transformer_cls.bool_list = bool_list
        transformer_cls.previous_residual = None
        transformer_cls.store = []
        transformer_cls.verbose = True
        print(f"[LeMiCa] enabled. cache={cache_name}, B={target_step}, computed={sum(bool_list)}/{len(bool_list)}")
    else:
        print("[LeMiCa] disabled. running baseline ErnieImage.")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        use_pe=args.use_pe,
    )
    latency = time.time() - start

    cache_tag = args.cache if args.cache is not None else "nocache"
    for idx, image in enumerate(output.images):
        image_path = os.path.join(args.output_dir, f"ernie_lemica_{cache_tag}_{idx}.png")
        image.save(image_path)
        print(f"[Save] {image_path}")
    print(f"[Done] latency={latency:.3f}s")


if __name__ == "__main__":
    main()
