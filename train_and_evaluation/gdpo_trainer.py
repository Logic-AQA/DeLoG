from typing import Any

import torch


def should_apply_official_gdpo(apply_gdpo: bool, reward_weights: torch.Tensor) -> bool:
    return bool(apply_gdpo) and reward_weights.numel() > 1


def compute_official_gdpo_advantages(
    rewards_per_func: torch.Tensor,
    reward_weights: torch.Tensor,
    *,
    num_generations: int,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute GDPO advantages following NVLabs' trainer patch."""
    if rewards_per_func.ndim != 2:
        raise ValueError("rewards_per_func must have shape (batch_times_generations, num_rewards)")
    if num_generations <= 1:
        raise ValueError("num_generations must be greater than 1 for GDPO")
    if rewards_per_func.shape[0] % num_generations != 0:
        raise ValueError("rewards_per_func rows must be divisible by num_generations")
    if reward_weights.numel() != rewards_per_func.shape[1]:
        raise ValueError(
            "reward_weights length must match the number of reward functions "
            f"({reward_weights.numel()} vs {rewards_per_func.shape[1]})"
        )

    rewards_per_func = torch.nan_to_num(rewards_per_func, nan=0.0)
    weights = reward_weights.to(device=rewards_per_func.device, dtype=rewards_per_func.dtype)
    grouped_rewards = rewards_per_func.view(-1, num_generations, rewards_per_func.shape[1])
    mean_grouped_rewards = grouped_rewards.mean(dim=1)
    std_grouped_rewards = grouped_rewards.std(dim=1)
    expanded_means = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    expanded_stds = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
    per_component_advantages = (rewards_per_func - expanded_means) / (expanded_stds + eps)
    pre_batch_norm_advantages = torch.sum(per_component_advantages * weights.unsqueeze(0), dim=1)
    advantages = (pre_batch_norm_advantages - pre_batch_norm_advantages.mean()) / (
        pre_batch_norm_advantages.std() + eps
    )

    details = {
        "per_component_advantages": per_component_advantages,
        "pre_batch_norm_advantages": pre_batch_norm_advantages,
        "component_group_stds": expanded_stds,
    }
    return advantages, details


def get_official_gdpo_trainer_class(base_cls=None):
    """Return a GRPOTrainer subclass with the NVLabs GDPO advantage path."""
    if base_cls is None:
        from trl import GRPOTrainer as base_cls

    from accelerate.utils import gather_object
    from trl.data_utils import is_conversational
    from trl.trainer.utils import nanmax, nanmin, nanstd, pad

    class OfficialGDPOTrainer(base_cls):
        def __init__(self, *args: Any, apply_gdpo: bool = False, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.apply_gdpo = bool(getattr(self.args, "apply_gdpo", apply_gdpo))

        def _generate_and_score_completions(self, inputs):
            device = self.accelerator.device
            mode = "train" if self.model.training else "eval"

            prompts = [x["prompt"] for x in inputs]

            if "images" in inputs[0]:
                images = [example.get("images") for example in inputs]
            elif "image" in inputs[0]:
                images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
            else:
                images = None
            if images is not None and all(img_list == [] for img_list in images):
                images = None

            (
                prompt_ids_list,
                completion_ids_list,
                num_items_in_batch,
                sampling_per_token_logps_list,
                forward_kwargs,
            ) = self._generate(prompts, images)

            prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
            prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
            prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
            prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
            completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
            if sampling_per_token_logps_list is not None:
                sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
                sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
            else:
                sampling_per_token_logps = None

            if self.mask_truncated_completions:
                eos_and_pad = [self.eos_token_id, self.pad_token_id]
                is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
                completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            if "token_type_ids" in forward_kwargs:
                token_type_ids = forward_kwargs["token_type_ids"]
                forward_kwargs["token_type_ids"] = torch.cat(
                    [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
                )

            logits_to_keep = completion_ids.size(1)
            batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
            num_images = [len(img_list) for img_list in images] if images is not None else None

            with torch.no_grad():
                generate_every = self.args.steps_per_generation * self.num_iterations
                if self.args.gradient_accumulation_steps % generate_every != 0 or (
                    self.use_vllm and self.vllm_importance_sampling_correction
                ):
                    old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
                else:
                    old_per_token_logps = None

                if self.use_vllm and self.vllm_importance_sampling_correction:
                    importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                    importance_sampling_ratio = torch.clamp(
                        importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )

                if self.beta != 0.0:
                    if self.ref_model is not None:
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.ref_model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,
                        )
                    else:
                        with self.accelerator.unwrap_model(self.model).disable_adapter():
                            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                num_images=num_images,
                                **forward_kwargs,
                            )
                else:
                    ref_per_token_logps = None

            prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
            else:
                completions = completions_text

            rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

            if should_apply_official_gdpo(self.apply_gdpo, self.reward_weights):
                advantages, gdpo_details = compute_official_gdpo_advantages(
                    rewards_per_func,
                    self.reward_weights,
                    num_generations=self.num_generations,
                )
                std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
            else:
                advantages = rewards - mean_grouped_rewards
                if self.scale_rewards in ["group", "none"]:
                    std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
                elif self.scale_rewards == "batch":
                    std_rewards = rewards.std().expand_as(rewards)
                else:
                    raise ValueError(
                        f"Invalid value for scale_rewards: {self.scale_rewards}. "
                        "Must be one of 'batch', 'group', or 'none'."
                    )
                if self.scale_rewards != "none":
                    advantages = advantages / (std_rewards + 1e-4)

            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            all_process_advantages = advantages.clone()
            advantages = advantages[process_slice]

            for i, reward_func_name in enumerate(self.reward_func_names):
                mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
                self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
                std_func_rewards = nanstd(rewards_per_func[:, i]).item()
                self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
            self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
            self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
            self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
            if self.apply_gdpo:
                self._metrics[mode]["gdpo/pre_batch_norm_advantage_std"].append(
                    gdpo_details["pre_batch_norm_advantages"].std().item()
                )
                self._metrics[mode]["gdpo/component_group_std"].append(
                    gdpo_details["component_group_stds"].mean().item()
                )

            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))
            for i, name in enumerate(self.reward_func_names):
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._logs["advantages"].extend(all_process_advantages.tolist())

            if images is not None:
                self._logs["images"].extend(gather_object(images))

            if self.use_vllm and self.vllm_importance_sampling_correction:
                delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
                delta = delta[completion_mask.bool()]
                mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
                max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
                self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                    self.accelerator.gather(mean_delta).mean().item()
                )
                self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                    self.accelerator.gather(max_delta).max().item()
                )

                flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
                min_importance_sampling_ratio = (
                    torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                mean_importance_sampling_ratio = (
                    torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                max_importance_sampling_ratio = (
                    torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                    nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                    self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
                )
                self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                    nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
                )

            output = {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "advantages": advantages,
                "num_items_in_batch": num_items_in_batch,
            }
            if old_per_token_logps is not None:
                output["old_per_token_logps"] = old_per_token_logps
            if self.use_vllm and self.vllm_importance_sampling_correction:
                output["importance_sampling_ratio"] = importance_sampling_ratio
            if ref_per_token_logps is not None:
                output["ref_per_token_logps"] = ref_per_token_logps
            if "pixel_values" in forward_kwargs:
                output["pixel_values"] = forward_kwargs["pixel_values"]
            if "image_grid_thw" in forward_kwargs:
                output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
            if "pixel_attention_mask" in forward_kwargs:
                output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
            if "image_sizes" in forward_kwargs:
                output["image_sizes"] = forward_kwargs["image_sizes"]
            if "token_type_ids" in forward_kwargs:
                output["token_type_ids"] = forward_kwargs["token_type_ids"]
            if images is not None:
                output["num_images"] = num_images
            return output

    return OfficialGDPOTrainer
