import torch
from diffusers import ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from typing import List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence
import argparse


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

def Lemica_call(
    self,
    x: List[torch.Tensor],
    t,
    cap_feats: List[torch.Tensor],
    patch_size=2,
    f_patch_size=1,
):
    assert patch_size in self.all_patch_size
    assert f_patch_size in self.all_f_patch_size

    bsz = len(x)
    device = x[0].device
    t = t * self.t_scale
    t = self.t_embedder(t)

    (
        x,
        cap_feats,
        x_size,
        x_pos_ids,
        cap_pos_ids,
        x_inner_pad_mask,
        cap_inner_pad_mask,
    ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

    # x embed & refine
    x_item_seqlens = [len(_) for _ in x]
    assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
    x_max_item_seqlen = max(x_item_seqlens)

    x = torch.cat(x, dim=0)
    x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

    # Match t_embedder output dtype to x for layerwise casting compatibility
    adaln_input = t.type_as(x)
    x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
    x = list(x.split(x_item_seqlens, dim=0))
    x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

    x = pad_sequence(x, batch_first=True, padding_value=0.0)
    x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for layer in self.noise_refiner:
            x = self._gradient_checkpointing_func(layer, x, x_attn_mask, x_freqs_cis, adaln_input)
    else:
        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

    # cap embed & refine
    cap_item_seqlens = [len(_) for _ in cap_feats]
    assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
    cap_max_item_seqlen = max(cap_item_seqlens)

    cap_feats = torch.cat(cap_feats, dim=0)
    cap_feats = self.cap_embedder(cap_feats)
    cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
    cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
    cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

    cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
    cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
    cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(cap_item_seqlens):
        cap_attn_mask[i, :seq_len] = 1

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for layer in self.context_refiner:
            cap_feats = self._gradient_checkpointing_func(layer, cap_feats, cap_attn_mask, cap_freqs_cis)
    else:
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

    # unified
    unified = []
    unified_freqs_cis = []
    for i in range(bsz):
        x_len = x_item_seqlens[i]
        cap_len = cap_item_seqlens[i]
        unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
        unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
    unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
    assert unified_item_seqlens == [len(_) for _ in unified]
    unified_max_item_seqlen = max(unified_item_seqlens)

    unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
    unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
    unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(unified_item_seqlens):
        unified_attn_mask[i, :seq_len] = 1

    
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
            unified += self.previous_residual
        else:
            ori_hidden_states = unified.clone()
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for layer in self.layers:
                    unified = self._gradient_checkpointing_func(
                        layer, unified, unified_attn_mask, unified_freqs_cis, adaln_input
                    )
            else:
                for layer in self.layers:
                    unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
    
            self.previous_residual = unified - ori_hidden_states
            
            
    unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
    unified = list(unified.unbind(dim=0))
    x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

    return x, {}


def get_args():
    parser = argparse.ArgumentParser(description="Run FLUX.1 with LeMiCa bool control.")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Enable caching: choose from [slow, medium, fast] or a numeric value. "
             "If omitted, caching is disabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = get_args()
    seed = args.seed
    
    num_inference_steps = 9
    
    speed_modes = {
        "slow": 8,
        "medium": 7,
        "fast": 6,
    }

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
            8: [0, 1, 2, 3, 5, 7, 8, 9],
            7: [0, 1, 2, 4, 6, 8, 9],
            6: [0, 1, 2, 5, 8, 9],
        }
        if lemica_step not in calc_dict:
            raise ValueError(f"cache step {lemica_step} not in calc_dict")

        calc_list = calc_dict[lemica_step]
        bool_list = [i in calc_list for i in range(num_inference_steps)]
    else:
        lemica_step = None
        bool_list = None

    pipeline = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipeline.to("cuda")

    if args.cache is not None:
        print("[INFO] Cache ENABLED")
        print(f"[INFO] Using cache: {args.cache} -> {lemica_step}")

        ZImageTransformer2DModel.forward = Lemica_call
        pipeline.transformer.__class__.enable_lemica = True
        pipeline.transformer.__class__.cnt = 0
        pipeline.transformer.__class__.num_steps = num_inference_steps
        pipeline.transformer.__class__.bool_list = bool_list
        pipeline.transformer.__class__.previous_residual = None
        pipeline.transformer.__class__.store = []
    else:
        print("[INFO] Cache DISABLED")

    print("[INFO] Model loaded.\n")

    prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    image = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]

    end.record()
    elapsed_time = start.elapsed_time(end) * 1e-3
    print(f"epoch time: {elapsed_time:.2f} sec")

    if args.cache is not None:
        image.save(f"example_{lemica_step}.png")
    else:
        image.save("example.png")


if __name__ == "__main__":
    main()