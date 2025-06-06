# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from pprint import pprint
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import time 
import os
import pandas as pd
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, AdvantageEstimator
from verl.trainer.ppo.metric_utils import (compute_data_metrics, compute_throughout_metrics, compute_timing_metrics,
                                           reduce_metrics)
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.dataset.rl_dataset import RLHFDataset
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

def add_prefix_in_dict(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}

def as_obj_vector(seq_of_lists):
    out = np.empty(len(seq_of_lists), dtype=object)
    for i, item in enumerate(seq_of_lists):
        out[i] = list(item)
    return out
def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        if (key == "metric_list" or key == "step_list"):
            non_tensors[key] = as_obj_vector(val)
        else:
            non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}

def sample_score(arr, n,beta=0.5, eps=0.1,mode="var"):
    #高分优先
    arr = np.asarray(arr, dtype=np.float32)
    if mode == "var":
        if len(arr) <2:
            return n
        var = np.var(arr) / (n**2) 
        return var   #优先算方差最大的
    elif mode == "max":
        return arr[-1]
    elif mode == "min":
        return -arr[-1]
    elif mode == "mean":
        if len(arr) <2:
            return n
        return - abs(n/2 - arr[-1])  #优先算reward最靠近中间的
    elif mode == "diff":
        # 学习 or 遗忘
        if len(arr) < 2:
            return n
        mean = np.mean(arr[:-1])
        diff = abs(arr[-1] - mean) #优先算能够使得方差变化最大的
        return diff
    else:
        var = np.var(arr) / (n**2) 
        inv_len = 1.0 / (len(arr) + eps)
        return (var + eps) ** beta * (inv_len + eps) ** (1 - beta)

class RayDAPOTrainer(RayPPOTrainer):
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type
        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                                f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset

        self.train_dataset = dataset_cls(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.get('gen_batch_size',
                                                                                   self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=False,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _create_new_dataloader(self, data_files):
        print("create train dataloader......")
        train_dataset = RLHFDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )


        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=train_dataset)

        return StatefulDataLoader(dataset=train_dataset,
                                batch_size=self.config.data.get('gen_batch_size',
                                                self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=False,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        sample_steps=0
        t0 = time.perf_counter()
        self.global_steps = 0
        batch_size=self.config.data.get('gen_batch_size',self.config.data.train_batch_size)
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        if self.config.trainer.get('enable_var_select', False):
            pid_var_dict ={}
        num_prompt_in_batch = 0
        num_gen_batches = 0

        #self.train_dataloader.next_iter_state = None
        trainer_dataloader = self.train_dataloader
        score_mode= self.config.trainer.get('score_mode', 'var')
        for epoch in range(self.config.trainer.total_epochs):
            if self.config.trainer.get('enable_dataset_update', False):
                epoch_records = []
            for batch_dict in trainer_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # last step and get a small batch size of test_size < batch_size, we append the batch to the epoch_records and use for the next epoch
                test_size = len(new_batch.non_tensor_batch['raw_prompt'])
                if test_size <batch_size:
                    print(f"last step and get a small batch size of {test_size} < {batch_size}")
                    if self.config.trainer.get('enable_dataset_update', False):
                        for idx in range(test_size):
                            epoch_records.append(
                                self._make_record_for_parquet(
                                    new_batch, idx, None,None
                                )
                            )
                    continue

                metrics = {}
                sample_steps    +=1
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]


                    new_batch.batch["response_mask"] = compute_response_mask(new_batch)
                    if score_mode.startswith("entropy"):
                        with _timer("entropy_select", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(new_batch)
                            entropys        = old_log_prob.batch["entropys"]        # [B, L]
                            response_masks  = new_batch.batch["response_mask"]  # [B, L] 0/1

                            ent_sum  = (entropys * response_masks).sum(dim=-1)                        # [B]
                            tok_cnt  = response_masks.sum(dim=-1).clamp(min=1)                        # [B]
                            avg_ent  = ent_sum / tok_cnt                                              # [B]

                            new_batch.non_tensor_batch["entropys_avg"] = avg_ent.cpu().numpy()


                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)


                        prompt_uid2metric_sum = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_sum[prompt_uid] = sum(metric_vals)
                        sum_max=self.config.trainer.get('sum_max', self.config.actor_rollout_ref.rollout.n-1)
                        sum_min=self.config.trainer.get('sum_min', 1)
                        kept_prompt_uids = [
                            uid for uid, prompt_sum in prompt_uid2metric_sum.items()
                            if (prompt_sum >= sum_min and 
                                prompt_sum <= sum_max) or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        if score_mode.startswith("entropy"):
                            prompt_entropys_avg_list = defaultdict(list)
                            for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch["entropys_avg"]):
                                prompt_entropys_avg_list[uid].append(metric_val)
                            prompt_entropys_metric = {}
                            if score_mode=="entropy_max":
                                for prompt_uid, entropys in prompt_entropys_avg_list.items():
                                    prompt_entropys_metric[prompt_uid] = sum(entropys) / len(entropys)
                            elif score_mode=="entropy_var":
                                for prompt_uid, entropys in prompt_entropys_avg_list.items():
                                    prompt_entropys_metric[prompt_uid] = np.var(entropys)


                        visit_prompt_uids = set()
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            metric_sum_val = prompt_uid2metric_sum[traj_from_prompt_uid]
                            if self.config.trainer.get('enable_dataset_update', False) and (traj_from_prompt_uid not in visit_prompt_uids):
                                epoch_records.append(
                                    self._make_record_for_parquet(
                                        new_batch, idx, metric_sum_val,self.global_steps
                                    )
                                )
                                visit_prompt_uids.add(traj_from_prompt_uid)
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                                if self.config.trainer.get('enable_var_select', False):
                                    if score_mode.startswith("entropy"):
                                        score = prompt_entropys_metric[traj_from_prompt_uid]
                                        pid_var_dict[traj_from_prompt_uid] = score
                                    else:
                                        tmp_list = list(new_batch.non_tensor_batch['metric_list'][idx])
                                        tmp_list.append(metric_sum_val)
                                        score = sample_score(tmp_list,n=self.config.actor_rollout_ref.rollout.n,mode=score_mode)
                                        pid_var_dict[traj_from_prompt_uid] = score
                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        sort_prompt_bsz = self.config.data.get('sort_prompt_bsz',prompt_bsz)
                        if num_prompt_in_batch < sort_prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {sort_prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            if self.config.trainer.get('enable_var_select', False):
                                n_roll=self.config.actor_rollout_ref.rollout.n
                                batch = self.var_select(batch,n=n_roll,dst_batch_size=prompt_bsz,pid_var_dict=pid_var_dict)
                            else:
                                traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
                                batch = batch[:traj_bsz]

                    # === Updating ===


                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
 
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                if self.config.trainer.get('enable_var_select', False):
                    pid_var_dict ={}

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                logger.log(data=add_prefix_in_dict(metrics, "sample_steps/"), step=sample_steps)
                wall_sec = int(time.perf_counter() - t0)
                logger.log(data=add_prefix_in_dict(metrics, "spend_time/"), step=wall_sec)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
            # update the train_dataloader for the next epoch
            if self.config.trainer.get('enable_dataset_update', False):     
                # 1. filter epoch_records
                dataset_metrics ={}
                filter_min = self.config.trainer.get('filter_min', 0)
                filter_max = self.config.trainer.get('filter_max', self.config.actor_rollout_ref.rollout.n)
                epoch_records,replay_buffer = self._filter_epoch_record(epoch_records, filter_min, filter_max)
                
                # 2. write epoch_records to parquet
                epoch_dir = self.config.trainer.get("default_local_dir", ".")
                os.makedirs(epoch_dir, exist_ok=True)
                epoch_file = os.path.join(epoch_dir, f"epoch_{epoch}_data.parquet")

                replay_file = os.path.join(epoch_dir, f"replay_buffer.parquet")
                local_replay_buffer = self._load_replay_buffer(replay_file)

                next_replay_buffer = local_replay_buffer + replay_buffer

                buffer_size = min(self.config.trainer.replay_buffer_size,len(next_replay_buffer))
                epoch_records = epoch_records + next_replay_buffer[:buffer_size]
                next_replay_buffer = next_replay_buffer[buffer_size:]

                print(f"get buffer size {buffer_size} records from replay buffer.")
                print(f"Left {len(epoch_records)} records after filtering.") 
                print(f"Left {len(next_replay_buffer)} records in replay buffer.")
                dataset_metrics["dataset/dataset_len"] = len(epoch_records)
                logger.log(data=dataset_metrics, step=self.global_steps-1)
                if len(epoch_records) == 0:
                    return 

                pd.DataFrame(next_replay_buffer).to_parquet(replay_file, index=False)
                pd.DataFrame(epoch_records).to_parquet(epoch_file, index=False) 

                # 3. update the train_dataloader
                trainer_dataloader=self._create_new_dataloader([epoch_file])
    def _make_record_for_parquet(self, batch: DataProto, idx: int, metric_sum_val,step) -> dict:
        record = {"metric_sum": metric_sum_val}
        for k, v in batch.non_tensor_batch.items():
            if k in  ('reward_model',"data_source","extra_info"):
                record[k] = v[idx]
            if k == "raw_prompt":
                record["prompt"] = v[idx].tolist()
            if k == "metric_list":
                tmp_list = list(v[idx])
                if metric_sum_val !=None:
                    tmp_list.append(metric_sum_val)
                record[k] = tmp_list
            if k == "step_list":
                tmp_list = list(v[idx])
                if step !=None:
                    tmp_list.append(step)
                record[k] = tmp_list
        return record
    def _filter_epoch_record(self, epoch_records: list[dict], filter_min,filter_max) -> list[dict]:
        filtered_records = []
        replay_buffer = []
        for record in epoch_records:
            if record['metric_sum'] == None or (record['metric_sum'] >=filter_min and record['metric_sum'] <= filter_max):
                filtered_records.append(record)
            else:
                replay_buffer.append(record)
        return filtered_records,replay_buffer
    
    def var_select(self, batch,n, dst_batch_size: int, pid_var_dict: dict):
        top_prompt_ids = [
            pid for pid, _ in sorted(
                pid_var_dict.items(), key=lambda kv: kv[1], reverse=True
            )[:dst_batch_size]
        ]
        selected_indices = [idx for idx, uid in enumerate(batch.non_tensor_batch['uid']) if uid in top_prompt_ids]
        assert dst_batch_size*n == len(selected_indices)
        new_batch = batch[selected_indices]
        return new_batch


    def _load_replay_buffer(self,path) -> list[dict]:
        if os.path.exists(path):
            return pd.read_parquet(path).to_dict(orient="records")
        return []