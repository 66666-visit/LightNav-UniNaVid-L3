import numpy as np

from copy import deepcopy

from habitat.core.agent import Agent
import contextlib
import json
import os
import re
import time
import torch
import cv2
import imageio
from habitat.utils.visualizations import maps
import random

from l3_integration.official_prompt import (
    build_prompt,
    build_task_text,
    prompt_sha256,
    verify_task_text_injection,
)

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


@contextlib.contextmanager
def _preserve_rng_state():
    """Keep the optional L3 perception path from perturbing model sampling.

    Both experiment groups are paired per Episode.  The L3 sidecar runs YOLO
    and serialization before ``model.generate``; neither may advance the RNG
    that the official decoder consumes.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    try:
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
    except Exception:  # noqa: BLE001 - CUDA may be unavailable in CPU tests
        cuda_states = None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class UniNaVid_Agent(Agent):
    def __init__(self, model_path, result_path, exp_save, l3_sidecar=None,
                 env=None, config=None, experiment_group="uninavid_baseline",
                 memory_backend=None, camera_spec=None,
                 memory_token_budget=True):
        
        print("Initialize UniNaVid")

        self.l3 = l3_sidecar
        self.memory = memory_backend
        self.env = env
        self.config = config
        self.camera_spec = camera_spec
        self.experiment_group = experiment_group
        self.memory_token_budget_enabled = bool(memory_token_budget)
        self.memory_prompt_mode = (
            getattr(getattr(memory_backend, "plan", None), "mode", None)
            if memory_backend is not None else None
        )
        if l3_sidecar is not None and memory_backend is not None:
            raise ValueError("use either the legacy L3 sidecar or the memory backend")

        self.result_path = result_path
        self.require_map = True if "video" in exp_save else False
        self.require_data = True if "data" in exp_save else False

        
        self.conv_mode = "vicuna_v1"
        
        os.makedirs(self.result_path, exist_ok=True)
        self.steps_path = os.path.join(self.result_path, "steps.jsonl")
        # Per-step diagnostics are only materialised for data-saving runs.
        self._steps_stream = (
            open(self.steps_path, "a", encoding="utf-8")
            if self.require_data else None
        )

        if self.require_data:
            os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        
        if self.require_data:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)


        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))


        print("Initialization Complete")

        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
        self.prompt_template_sha256 = prompt_sha256(self.promt_template)

        # self.history_rgb_tensor = None
        
        self.rgb_list = []
        self.topdown_map_list = []
        
        self.count_id = 0
        self.reset()


    def process_images(self, rgb_list):
        
        # start_img_index = 0
        
        # if self.history_rgb_tensor is not None:
            # start_img_index = self.history_rgb_tensor.shape[0]
        
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        # if self.history_rgb_tensor is None:
        #     self.history_rgb_tensor = video
        # else:
        #     self.history_rgb_tensor = torch.cat((self.history_rgb_tensor, video), dim = 0)
        # rgb_list = []

        return [video]


    #: Official generation parameters, kept in one place so the token budget
    #: reserves exactly what ``model.generate`` is allowed to emit.
    OFFICIAL_MAX_NEW_TOKENS = 1024
    OFFICIAL_TEMPERATURE = 0.2

    def _assemble_input_ids(self, prompt, *, cuda=True):
        """Official Uni-NaVid token assembly, unchanged except for the device.

        The same method is used by ``predict_inference`` (CUDA) and by the
        read-only token-budget measurement (CPU).  No token is added, removed
        or reordered relative to upstream ``predict_inference``.
        """
        def on_device(tensor):
            return tensor.cuda() if cuda else tensor

        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = on_device(self.tokenizer(
            IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:])
        image_end_special_token = on_device(self.tokenizer(
            IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:])
        video_start_special_token = on_device(self.tokenizer(
            VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:])
        video_end_special_token = on_device(self.tokenizer(
            VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:])
        navigation_special_token = on_device(self.tokenizer(
            NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:])
        image_seperator = on_device(self.tokenizer(
            IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:])

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = on_device(tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'))
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)
        return input_ids, question

    def measure_prompt_tokens(self, prompt):
        """Return the exact official input-id count for one prepared prompt."""
        with torch.inference_mode():
            input_ids, _question = self._assemble_input_ids(prompt, cuda=False)
        return int(input_ids.shape[1])

    def predict_inference(self, prompt):
        input_ids, question = self._assemble_input_ids(prompt, cuda=True)
        self.last_prompt_tokens = int(input_ids.shape[1])

        stop_str = (
            conv_templates[self.conv_mode].sep
            if conv_templates[self.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[self.conv_mode].sep2
        )
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=self.OFFICIAL_TEMPERATURE,
                max_new_tokens=self.OFFICIAL_MAX_NEW_TOKENS,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return None, None
            match = match.group()
            return 3, float(match)

        return None, None



    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]



        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line


        if line:

            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)


        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image


    def reset(self, episode=None, instruction=""):
                
        if self.require_map:
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))

                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0
        self.first_forward = False

        # --- L3 instrumentation/state (does not alter the official loop) ---
        self.step_index = 0
        self.step_diagnostics = []
        self.inference_latencies_ms = []
        self.prompt_token_counts = []
        self.last_prompt_tokens = None
        self.last_inference_ms = None
        self.last_prompt_text = None
        self.last_task_text = None
        self.last_l3_context_text = None
        self.last_memory_context_text = None
        self.last_memory_diagnostics = None
        self.last_memory_token_budget = None
        self.last_memory_frame = None
        self.last_executed_action = None
        self.last_prediction = None
        self.episode_instruction = instruction
        # Per-module timings, one list per step; summary() reports their mean.
        self.memory_observe_ms = []
        self.memory_l3_observe_ms = []
        self.memory_retrieval_ms = []
        self.memory_prompt_build_ms = []
        self.memory_token_budget_ms = []
        self.memory_injection_flags = []
        self.memory_omitted_items = []
        self.memory_token_overflow_drops = []
        self.memory_duplicates_removed = []
        self.memory_conflict_counts = []
        self.memory_l1_fact_counts = []
        self.memory_l3_fact_counts = []
        self.memory_retrieved_node_counts = []
        self.memory_evidence_ids = []
        memory = getattr(self, "memory", None)
        if memory is not None:
            if episode is None:
                with _preserve_rng_state():
                    memory.close_episode()
            else:
                if not self.episode_instruction:
                    self.episode_instruction = _episode_instruction(episode)
                with _preserve_rng_state():
                    memory.reset_episode(episode, self.episode_instruction)
        if getattr(self, "l3", None) is not None:
            if episode is None:
                # Construction calls reset() before an Episode exists.  Do not
                # start a map; the first per-Episode reset supplies the Episode.
                with _preserve_rng_state():
                    self.l3.close_episode()
            else:
                if not self.episode_instruction:
                    self.episode_instruction = _episode_instruction(episode)
                with _preserve_rng_state():
                    self.l3.reset_episode(episode, self.episode_instruction)


    def _observe_l3(self, observations, step):
        sidecar = getattr(self, "l3", None)
        if sidecar is None:
            return {"l3_enabled": False}
        diagnostic = {"l3_enabled": True, "agent_state_error": None}
        agent_state = None
        env = getattr(self, "env", None)
        if env is None:
            diagnostic["agent_state_error"] = "navigation env not bound"
        else:
            try:
                agent_state = env.sim.get_agent_state()
            except Exception as exc:  # noqa: BLE001 - report, never invent a pose
                diagnostic["agent_state_error"] = f"{type(exc).__name__}: {exc}"
        try:
            with _preserve_rng_state():
                observation_diagnostic = sidecar.observe(
                    observations, agent_state, step,
                    transition_record=bool(step > 0),
                )
        except Exception as exc:  # noqa: BLE001 - optional sidecar fails closed
            observation_diagnostic = {
                "step": step, "frame_valid": False,
                "frame_error": f"{type(exc).__name__}: {exc}",
            }
        if isinstance(observation_diagnostic, dict):
            diagnostic.update(observation_diagnostic)
        else:
            diagnostic["frame_valid"] = False
            diagnostic["frame_error"] = "invalid L3 observe diagnostic"
        return diagnostic


    def _agent_state(self):
        env = getattr(self, "env", None)
        if env is None:
            return None, "navigation env not bound"
        try:
            return env.sim.get_agent_state(), None
        except Exception as exc:  # noqa: BLE001 - report, never invent a pose
            return None, f"{type(exc).__name__}: {exc}"

    def _camera_spec(self):
        if self.camera_spec is not None:
            return self.camera_spec
        if self.config is None:
            raise ValueError("L3 frame construction needs a Habitat config")
        from l3_integration.habitat_geometry import camera_spec_from_config

        self.camera_spec = camera_spec_from_config(self.config)
        return self.camera_spec

    def _build_frame(self, observations):
        """Build the official RGB-D frame from the current observation only."""
        state, state_error = self._agent_state()
        if state is None:
            return None, state_error
        try:
            from l3_integration.habitat_geometry import frame_from_habitat

            frame, metadata = frame_from_habitat(
                observations, state, self._camera_spec(),
            )
        except Exception as exc:  # noqa: BLE001 - optional L3 fails closed
            return None, f"{type(exc).__name__}: {exc}"
        if not metadata.get("frame_valid", True):
            return None, metadata.get("frame_error", "invalid RGB-D frame")
        return frame, None

    def _observe_memory(self, observations, step):
        """Write one observation into the L1 writer and, if enabled, the L3 map."""
        memory = getattr(self, "memory", None)
        if memory is None:
            return {"memory_enabled": False}
        diagnostic = {
            "memory_enabled": True,
            "l1_enabled": bool(memory.plan.uses_l1),
            "l3_enabled": bool(memory.plan.uses_l3),
            "events": [],
            "frame": None,
        }
        try:
            with _preserve_rng_state():
                events, observe_diagnostic = memory.observe(
                    observations, step,
                    last_action=self.last_executed_action,
                    prediction=self.last_prediction,
                )
        except Exception as exc:  # noqa: BLE001 - optional memory fails closed
            diagnostic["memory_error"] = f"{type(exc).__name__}: {exc}"
            return diagnostic
        diagnostic["events"] = list(events)
        if isinstance(observe_diagnostic, dict):
            diagnostic.update(observe_diagnostic)
        if memory.plan.uses_l3:
            frame, frame_error = self._build_frame(observations)
            if frame is None:
                diagnostic["l3_frame_error"] = frame_error
            else:
                try:
                    with _preserve_rng_state():
                        spatial = memory.observe_spatial(
                            frame, events, step,
                            transition_record=bool(step > 0),
                        )
                except Exception as exc:  # noqa: BLE001 - optional L3 fails closed
                    spatial = {
                        "frame_valid": False,
                        "rejection_reason": f"{type(exc).__name__}: {exc}",
                    }
                diagnostic.update(spatial)
                diagnostic["frame"] = frame if spatial.get("frame_valid") else None
        self.last_memory_frame = diagnostic.get("frame")
        return diagnostic

    def _measure_memory_token_budget(self, instruction):
        """Measure the official prompt headroom for one memory block.

        The base prompt is the unchanged official template with the original
        instruction; its token count is produced by the same official assembly
        used by ``predict_inference``.  The generation reserve is the official
        ``max_new_tokens``.  If any measurement is unavailable the caller falls
        back to the original character budget and records why.
        """
        from l1_memory.token_budget import (
            BOUNDARY_RESERVE_TOKENS,
            MemoryTokenBudget,
            TokenBudgetFailure,
        )

        if not self.memory_token_budget_enabled:
            return None, TokenBudgetFailure(reason="token_budget_disabled_by_config")
        try:
            context_limit = getattr(self, "context_len", None)
            if type(context_limit) is not int or context_limit <= 0:
                return None, TokenBudgetFailure(
                    reason="context_limit_unavailable: context_len is not a positive integer",
                )
            base_prompt = build_prompt(self.promt_template, instruction)
            base_tokens = self.measure_prompt_tokens(base_prompt)
            reserve = self.OFFICIAL_MAX_NEW_TOKENS
            encode = getattr(self.tokenizer, "encode", None)
            if not callable(encode):
                return None, TokenBudgetFailure(
                    reason="tokenizer_encode_unavailable",
                    context_limit_tokens=context_limit,
                    base_prompt_tokens=base_tokens,
                    generation_reserve_tokens=reserve,
                )

            def counter(text):
                return len(encode(text, add_special_tokens=False)) + 1

            available = (
                context_limit - base_tokens - reserve - BOUNDARY_RESERVE_TOKENS
            )
            return MemoryTokenBudget(
                available_tokens=max(0, available),
                context_limit_tokens=context_limit,
                base_prompt_tokens=base_tokens,
                generation_reserve_tokens=reserve,
                safety_margin_tokens=BOUNDARY_RESERVE_TOKENS,
                context_limit_source="uninavid.model.builder.context_len",
                counter=counter,
            ), None
        except Exception as exc:  # noqa: BLE001 - report, never guess
            return None, TokenBudgetFailure(
                reason=f"token_budget_measurement_failed: {type(exc).__name__}: {exc}",
            )

    def _build_memory_prompt(self, instruction, step, observations):
        """Return (open-memory diagnostic, context result, task text)."""
        memory = getattr(self, "memory", None)
        if memory is None:
            return {"memory_enabled": False}, None, instruction
        diagnostic = {"memory_enabled": True, "agent_state_error": None}
        state, state_error = self._agent_state()
        if state_error:
            diagnostic["agent_state_error"] = state_error
        rgb = observations.get("rgb") if isinstance(observations, dict) else None
        rgb_width = int(rgb.shape[1]) if getattr(rgb, "ndim", 0) == 3 else 0
        if rgb_width < 1:
            diagnostic["memory_error"] = "current RGB width is unavailable"
            memory.discard_context()
            return diagnostic, None, instruction
        budget_started = time.perf_counter()
        budget, budget_failure = self._measure_memory_token_budget(instruction)
        diagnostic["token_budget_ms"] = (
            time.perf_counter() - budget_started
        ) * 1000.0
        self.last_memory_token_budget = budget
        diagnostic["token_budget"] = (
            budget.diagnostics() if budget is not None
            else budget_failure.diagnostics() if budget_failure is not None
            else {"budget_source": "char_fallback_320"}
        )
        started = time.perf_counter()
        try:
            with _preserve_rng_state():
                result = memory.build_context(
                    instruction=instruction, step=step, rgb_width=rgb_width,
                    frame=getattr(self, "last_memory_frame", None),
                    token_budget=budget, budget_failure=budget_failure,
                )
        except Exception as exc:  # noqa: BLE001 - experimental formatter fails closed
            memory.discard_context()
            diagnostic["memory_context_error"] = f"{type(exc).__name__}: {exc}"
            diagnostic["prompt_build_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            return diagnostic, None, instruction
        diagnostic["prompt_build_ms"] = (time.perf_counter() - started) * 1000.0
        if isinstance(result.diagnostics, dict):
            diagnostic.update(deepcopy(result.diagnostics))
        contexts = getattr(result, "context", None)
        text = contexts.text if contexts is not None else None
        if not isinstance(text, str) or not text.strip():
            text = None
        diagnostic["memory_context_text"] = text
        diagnostic["memory_source"] = getattr(result, "source", None)
        diagnostic["memory_prompt_mode"] = getattr(result.plan, "mode", self.memory_prompt_mode)
        task_text = build_task_text(instruction, text) if text else instruction
        return diagnostic, result, task_text

    def _count_tokens(self, text):
        try:
            return int(len(self.tokenizer.encode(text, add_special_tokens=False)))
        except Exception:  # noqa: BLE001 - token accounting is diagnostic only
            return None


    def _augment_task_text(self, instruction, step, diagnostic):
        """Return (task_text, l3_context_or_none, diagnostic).

        L3 is inserted only inside the official template's ``{}`` task slot.
        The complete official prompt is formatted *after* this composition;
        this method never appends to the full prompt.
        """
        sidecar = getattr(self, "l3", None)
        if sidecar is None:
            self.last_l3_context_text = None
            return instruction, None, diagnostic
        if not isinstance(instruction, str):
            # Keep the baseline path's formatting/verification behaviour; do
            # not let a malformed instruction become an L3-only failure.
            self.last_l3_context_text = None
            return instruction, None, diagnostic
        try:
            with _preserve_rng_state():
                organization, context_diagnostic = sidecar.build_context(
                    instruction, step,
                )
        except Exception as exc:  # noqa: BLE001 - optional L3 fails closed
            diagnostic["context_injected"] = False
            diagnostic["prompt_has_l3"] = False
            diagnostic["context_reason"] = f"{type(exc).__name__}: {exc}"
            self.last_l3_context_text = None
            return instruction, None, diagnostic
        def _no_context(reason):
            diagnostic["context_injected"] = False
            diagnostic["prompt_has_l3"] = False
            diagnostic["context_reason"] = reason
            self.last_l3_context_text = None
            return instruction, None, diagnostic

        if isinstance(context_diagnostic, dict):
            diagnostic.update(context_diagnostic)
        else:
            return _no_context("invalid L3 diagnostic")

        try:
            organization_context = getattr(organization, "context", None)
            context_text = (
                getattr(organization_context, "text", None)
                if organization_context is not None else None
            )
        except Exception as exc:  # noqa: BLE001 - malformed optional object
            return _no_context(f"{type(exc).__name__}: {exc}")
        if organization_context is None:
            return _no_context("no L3 context")
        if not isinstance(context_text, str) or not context_text.strip():
            # An optional memory channel fails closed: never inject a malformed
            # or empty block, and never touch the official task text.
            return _no_context("empty or invalid L3 context")
        task_text = build_task_text(instruction, context_text)
        diagnostic["context_injected"] = True
        diagnostic["prompt_has_l3"] = True
        diagnostic["context_tokens"] = self._count_tokens(context_text)
        self.last_l3_context_text = context_text
        return task_text, context_text, diagnostic


    @staticmethod
    def _json_default(value):
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        return str(value)


    def _finish_step(self, diagnostic, *, action, inference_ms, prompt_tokens,
                     prompt_text):
        row = dict(diagnostic)
        row.pop("events", None)
        row.pop("frame", None)
        row["episode_id"] = getattr(self, "episode_id", None)
        row["experiment_group"] = getattr(self, "experiment_group", None)
        row["step"] = self.step_index
        row["action"] = action
        row["inference_ms"] = inference_ms
        row["prompt_tokens"] = prompt_tokens
        row["prompt_sha256"] = (
            prompt_sha256(prompt_text) if prompt_text is not None else None
        )
        row["official_prompt_sha256"] = self.prompt_template_sha256
        if prompt_text is not None and os.environ.get("L3_LOG_PROMPTS") == "1":
            # Opt-in audit trail: the exact text passed to predict_inference.
            row["prompt_text"] = prompt_text
            row["task_text"] = self.last_task_text
            row["l3_context_text"] = self.last_l3_context_text
            row["memory_context_text"] = self.last_memory_context_text
        memory = getattr(self, "memory", None)
        row.setdefault("memory_prompt_mode", getattr(
            self, "memory_prompt_mode", None,
        ))
        row.setdefault("l3_enabled", (
            getattr(self, "l3", None) is not None
            or bool(memory is not None and memory.plan.uses_l3)
        ))
        row.setdefault("context_injected", False)
        row.setdefault("prompt_has_l3", False)
        row.setdefault("prompt_has_l1", False)
        row.setdefault("fact_count", 0)
        row.setdefault("context_chars", 0)
        row.setdefault("context_tokens", 0)
        row.setdefault("frame_valid", False)
        row.setdefault("frame_error", None)
        # Keep the original diagnostic aliases populated for the existing
        # episode summary, even though the memory backend reports finer fields.
        l1_observe = float(row.get("l1_observe_ms", 0.0) or 0.0)
        l3_observe = float(row.get("l3_observe_ms", 0.0) or 0.0)
        row.setdefault("segmentation_ms", l3_observe or None)
        row.setdefault("observe_ms", l1_observe + l3_observe)
        row.setdefault("retrieval_ms", row.get("prompt_build_ms"))
        row.setdefault("context_tokens", self._count_tokens(
            row.get("memory_context_text") or ""
        ))
        if row.get("memory_context_text") and not row.get("context_chars"):
            row["context_chars"] = len(row["memory_context_text"])
        self.step_diagnostics.append(row)
        stream = getattr(self, "_steps_stream", None)
        if stream is not None and not stream.closed:
            stream.write(json.dumps(row, ensure_ascii=False,
                                    default=self._json_default) + "\n")
            stream.flush()
        if inference_ms is not None:
            self.inference_latencies_ms.append(float(inference_ms))
        if prompt_tokens is not None:
            self.prompt_token_counts.append(int(prompt_tokens))
        if memory is not None:
            if inference_ms is not None:
                self.memory_injection_flags.append(
                    bool(row.get("context_injected"))
                )
                self.memory_omitted_items.append(
                    int(row.get("omitted_items", 0) or 0)
                )
                self.memory_token_overflow_drops.append(
                    int(row.get("token_overflow_drops", 0) or 0)
                )
                self.memory_duplicates_removed.append(
                    int(row.get("duplicates_removed", 0) or 0)
                )
                self.memory_conflict_counts.append(
                    int(row.get("conflict_count", 0) or 0)
                )
                self.memory_l1_fact_counts.append(
                    int(row.get("l1_fact_count", 0) or 0)
                )
                self.memory_l3_fact_counts.append(
                    int(row.get("l3_fact_count", 0) or 0)
                )
                self.memory_retrieved_node_counts.append(
                    int(row.get("retrieved_nodes", 0) or 0)
                )
                self.memory_evidence_ids.extend(
                    list(row.get("evidence_ids") or [])
                )
                self.memory_prompt_build_ms.append(
                    float(row["prompt_build_ms"])
                    if row.get("prompt_build_ms") is not None else None
                )
                self.memory_retrieval_ms.append(
                    float(row["retrieval_ms"])
                    if row.get("retrieval_ms") is not None else None
                )
                self.memory_token_budget_ms.append(
                    float(row["token_budget_ms"])
                    if row.get("token_budget_ms") is not None else None
                )
            self.memory_observe_ms.append(l1_observe)
            self.memory_l3_observe_ms.append(l3_observe)
        self.step_index += 1


    def episode_summary(self):
        rows = list(getattr(self, "step_diagnostics", []))
        l3_rows = [row for row in rows if row.get("l3_enabled")]
        context_rows = [row for row in l3_rows if row.get("context_injected")]
        seg_values = [
            float(row["segmentation_ms"]) for row in l3_rows
            if row.get("segmentation_ms") is not None
        ]
        observe_values = [
            float(row["observe_ms"]) for row in l3_rows
            if row.get("observe_ms") is not None
        ]
        retrieval_values = [
            float(row["retrieval_ms"]) for row in l3_rows
            if row.get("retrieval_ms") is not None
        ]
        total_values = [
            float(row.get("observe_ms", 0.0) or 0.0)
            + float(row.get("retrieval_ms", 0.0) or 0.0)
            for row in l3_rows
        ]
        context_tokens = [
            int(row["context_tokens"]) for row in context_rows
            if row.get("context_tokens") is not None
        ]
        fact_counts = [int(row.get("fact_count", 0) or 0) for row in context_rows]
        latencies = list(getattr(self, "inference_latencies_ms", []))
        prompt_tokens = list(getattr(self, "prompt_token_counts", []))
        sidecar = getattr(self, "l3", None)
        memory = getattr(self, "memory", None)
        map_summary = None
        if sidecar is not None:
            with _preserve_rng_state():
                map_summary = sidecar.map_snapshot()
        elif memory is not None:
            try:
                map_summary = memory.map_snapshot()
            except Exception:  # noqa: BLE001 - diagnostic only
                map_summary = None
        memory_summary = None
        if memory is not None:
            try:
                memory_summary = memory.summary()
            except Exception:  # noqa: BLE001 - diagnostic only
                memory_summary = None
        env = getattr(self, "env", None)
        scene_id = None
        if env is not None and getattr(env, "current_episode", None) is not None:
            scene_id = getattr(env.current_episode, "scene_id", None)
        return {
            "episode_id": getattr(self, "episode_id", None),
            "scene_id": scene_id,
            "experiment_group": getattr(self, "experiment_group", None),
            "l3_enabled": sidecar is not None,
            "env_steps": len(rows),
            "inference_calls": len(latencies),
            "inference_latency_ms_mean": _mean(latencies),
            "inference_latency_ms_p50": _percentile(latencies, 0.50),
            "inference_latency_ms_p95": _percentile(latencies, 0.95),
            "prompt_tokens_mean": _mean(prompt_tokens),
            "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
            "prompt_has_l3_steps": sum(bool(row.get("prompt_has_l3")) for row in rows),
            "l3_map_steps": sum(bool(row.get("frame_valid")) for row in l3_rows),
            "l3_frame_unavailable_steps": sum(
                not bool(row.get("frame_valid")) for row in l3_rows
            ),
            "l3_context_steps": len(context_rows),
            "l3_fact_count_total": sum(fact_counts),
            "l3_fact_count_mean_per_injected_step": _mean(fact_counts),
            "l3_context_chars_mean": _mean([
                float(row.get("context_chars", 0.0) or 0.0) for row in context_rows
            ]),
            "l3_context_tokens_mean": _mean(context_tokens),
            "l3_segmentation_ms_mean": _mean(seg_values),
            "l3_observe_ms_mean": _mean(observe_values),
            "l3_retrieval_ms_mean": _mean(retrieval_values),
            "l3_build_ms_mean": _mean(total_values),
            "l3_object_nodes_final": (
                map_summary["objects"] if map_summary is not None else 0
            ),
            "l3_map_summary": map_summary,
            "official_prompt_sha256": getattr(self, "prompt_template_sha256", None),
            "memory_enabled": memory is not None,
            "memory_prompt_mode": getattr(self, "memory_prompt_mode", None),
            "memory_activation": (
                memory_summary.get("activation") if memory_summary else None
            ),
            "memory_summary": memory_summary,
            "prompt_injection_rate": _mean(
                getattr(self, "memory_injection_flags", [])
            ),
            "prompt_injection_steps": sum(
                bool(flag) for flag in getattr(self, "memory_injection_flags", [])
            ),
            "prompt_injection_opportunities": len(
                getattr(self, "memory_injection_flags", [])
            ),
            "l1_fact_count_total": sum(
                getattr(self, "memory_l1_fact_counts", [])
            ),
            "l1_fact_count_mean": _mean(
                getattr(self, "memory_l1_fact_counts", [])
            ),
            "l3_fact_count_total_memory": sum(
                getattr(self, "memory_l3_fact_counts", [])
            ),
            "l3_fact_count_mean": _mean(
                getattr(self, "memory_l3_fact_counts", [])
            ),
            "duplicates_removed_total": sum(
                getattr(self, "memory_duplicates_removed", [])
            ),
            "conflict_count_total": sum(
                getattr(self, "memory_conflict_counts", [])
            ),
            "omitted_items_total": sum(
                getattr(self, "memory_omitted_items", [])
            ),
            "token_overflow_drops_total": sum(
                getattr(self, "memory_token_overflow_drops", [])
            ),
            "retrieved_nodes_mean": _mean(
                getattr(self, "memory_retrieved_node_counts", [])
            ),
            "retrieved_nodes_max": (
                max(self.memory_retrieved_node_counts)
                if getattr(self, "memory_retrieved_node_counts", []) else None
            ),
            "injected_evidence_id_count": len(set(
                getattr(self, "memory_evidence_ids", [])
            )),
            "l1_observe_ms_mean": _mean(getattr(self, "memory_observe_ms", [])),
            "l3_observe_ms_mean_memory": _mean(
                getattr(self, "memory_l3_observe_ms", [])
            ),
            "prompt_build_ms_mean": _mean(
                getattr(self, "memory_prompt_build_ms", [])
            ),
            "token_budget_ms_mean": _mean(
                getattr(self, "memory_token_budget_ms", [])
            ),
            "retrieval_ms_mean_memory": _mean(
                getattr(self, "memory_retrieval_ms", [])
            ),
        }


    def close(self):
        stream = getattr(self, "_steps_stream", None)
        if stream is not None and not stream.closed:
            stream.flush()
            stream.close()
        sidecar = getattr(self, "l3", None)
        if sidecar is not None:
            with _preserve_rng_state():
                sidecar.close_episode()
        memory = getattr(self, "memory", None)
        if memory is not None:
            with _preserve_rng_state():
                memory.close_episode()


    def _record_memory_context(self, result, context, diagnostic):
        """Copy the mode's auditable counts into the per-step row."""
        details = getattr(result, "diagnostics", None) if result is not None else None
        details = details if isinstance(details, dict) else {}
        layers = details.get("layers") or {}
        l1 = layers.get("l1") or {}
        l3 = layers.get("l3") or {}
        joint = layers.get("joint") or {}
        evidence_ids = list(details.get("evidence_ids") or [])
        if context is not None:
            evidence_ids = list(dict.fromkeys(
                ref.event_id for item in context.layer_items
                for ref in item.evidence_refs
            )) or evidence_ids
        alignment = details.get("alignment") or {}
        conflicts = details.get("conflicts") or []
        duplicates = int(details.get("duplicates_removed", 0) or 0)
        if not duplicates:
            duplicates = int(l1.get("duplicates_removed", 0) or 0) + int(
                l3.get("duplicates_removed", 0) or 0
            )
        omitted = int(details.get("omitted_items", 0) or 0)
        if not omitted:
            omitted = int(l1.get("omitted_items", 0) or 0) + int(
                l3.get("omitted_items", 0) or 0
            )
        map_summary = {}
        memory = getattr(self, "memory", None)
        if memory is not None:
            try:
                map_summary = memory.map_snapshot()
            except Exception:  # noqa: BLE001 - diagnostic only
                map_summary = {}
        selected = list(context.layer_items) if context is not None else []
        l1_selected = [item for item in selected if _item_has_l1(item)]
        l3_selected = [item for item in selected if _item_has_l3(item)]
        l1_evidence_ids = list(dict.fromkeys(
            evidence_id
            for item in l1_selected
            for evidence_id in (
                item.metadata.get("l1_evidence_ids")
                or [ref.event_id for ref in item.evidence_refs]
            )
        ))
        l3_evidence_ids = list(dict.fromkeys(
            evidence_id
            for item in l3_selected
            for evidence_id in (
                item.metadata.get("l3_evidence_ids")
                or [ref.event_id for ref in item.evidence_refs]
            )
        ))
        diagnostic.update({
            "mode": details.get("mode", getattr(
                getattr(result, "plan", None), "mode", self.memory_prompt_mode,
            )),
            "memory_source": details.get("source"),
            "l1_fact_count": len(l1_selected) if context is not None
            else int(l1.get("fact_count", l1.get("item_count", 0)) or 0),
            "l1_item_count": len(l1_selected) if context is not None
            else int(l1.get("item_count", 0) or 0),
            "l1_current_count": int(l1.get("current_count", 0) or 0),
            "l1_history_count": int(l1.get("history_count", 0) or 0),
            "l3_fact_count": len(l3_selected) if context is not None
            else int(l3.get("item_count", 0) or 0),
            "l3_candidate_count": int(l3.get("candidate_count", 0) or 0),
            "l3_verified_count": int(l3.get("verified_count", 0) or 0),
            "l3_unverified_count": int(l3.get("unverified_count", 0) or 0),
            "joint_item_count": len(selected) if context is not None
            else int(joint.get("item_count", 0) or 0),
            "joint_aligned_pairs": int(
                alignment.get("confirmed", joint.get("aligned_pairs", 0)) or 0
            ),
            "joint_attempted_pairs": int(alignment.get("attempted", 0) or 0),
            "joint_alignment_success_rate": alignment.get("success_rate"),
            "duplicates_removed": duplicates,
            "conflict_count": len(conflicts),
            "omitted_items": omitted,
            "retrieved_nodes": int(
                map_summary.get("objects", 0) or 0
                if l3 else (l1.get("fact_count", 0) or 0)
            ),
            "evidence_ids": evidence_ids,
            "l1_evidence_ids": l1_evidence_ids,
            "l3_evidence_ids": l3_evidence_ids,
            "l1_fact_ids": list(l1.get("items") or []),
            "l3_fact_ids": list(l3.get("items") or []),
            "context_chars": len(context.text) if context is not None else 0,
            "fact_count": len(selected) if context is not None else 0,
        })

    def _fit_memory_prompt(self, instruction, result, diagnostic):
        """Apply the exact token-delta check and whole-line trimming protocol.

        Returns ``(context, task_text, memory_tokens_injected,
        token_overflow_drops)``.  The official template and the original
        instruction are reformatted for every candidate; only the memory block
        inside the task slot changes.
        """
        from l1_memory.token_budget import trim_context_for_overflow

        if result is None:
            return None, instruction, 0, 0
        context = getattr(result, "context", None)
        if context is None:
            return None, instruction, 0, 0
        budget = self.last_memory_token_budget
        base_prompt = build_prompt(self.promt_template, instruction)
        try:
            base_tokens = self.measure_prompt_tokens(base_prompt)
        except Exception as exc:  # noqa: BLE001 - measurement failure only
            diagnostic["prompt_fallback_reason"] = (
                f"token_delta_measurement_failed: {type(exc).__name__}: {exc}"
            )
            base_tokens = None
        dropped_total = 0
        while context is not None:
            memory_text = context.text
            if not isinstance(memory_text, str) or not memory_text.strip():
                return None, instruction, 0, dropped_total
            task_text = build_task_text(instruction, memory_text)
            prompt = build_prompt(self.promt_template, task_text)
            verify_task_text_injection(
                self.promt_template, instruction, memory_text, prompt,
            )
            if budget is None or base_tokens is None:
                return context, task_text, None, dropped_total
            try:
                delta = self.measure_prompt_tokens(prompt) - base_tokens
            except Exception as exc:  # noqa: BLE001
                diagnostic["prompt_fallback_reason"] = (
                    f"token_delta_measurement_failed: {type(exc).__name__}: {exc}"
                )
                return context, task_text, None, dropped_total
            if delta <= budget.available_tokens:
                return context, task_text, delta, dropped_total
            new_context, dropped = trim_context_for_overflow(context)
            dropped_total += dropped
            if new_context is None or dropped == 0 or not new_context.layer_items:
                diagnostic["prompt_fallback_reason"] = (
                    "token budget overflow: no lower-priority fact item could "
                    "be dropped"
                )
                return None, instruction, 0, dropped_total
            context = new_context
        return None, instruction, 0, dropped_total

    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)
        instruction = observations["instruction"]["text"]
        step = self.step_index

        memory = getattr(self, "memory", None)
        if memory is not None:
            l3_diagnostic = self._observe_memory(observations, step)
            l3_diagnostic["step"] = step
        else:
            l3_diagnostic = self._observe_l3(observations, step)

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, instruction, "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            self.last_executed_action = {"action": temp_action}
            self._finish_step(
                l3_diagnostic, action=temp_action, inference_ms=None,
                prompt_tokens=None, prompt_text=None,
            )
            return {"action": temp_action}

        result = None
        if memory is not None:
            memo_diagnostic, result, task_text = self._build_memory_prompt(
                instruction, step, observations,
            )
            memo_diagnostic["step"] = step
            context, task_text, memory_tokens, drops = self._fit_memory_prompt(
                instruction, result, memo_diagnostic,
            )
            memory_text = context.text if context is not None else None
            navigation_qs = build_prompt(self.promt_template, task_text)
            verify_task_text_injection(
                self.promt_template, instruction, memory_text, navigation_qs,
            )
            memo_diagnostic["memory_context_text"] = memory_text
            memo_diagnostic["context_injected"] = context is not None
            memo_diagnostic["prompt_has_l3"] = bool(
                context is not None and memory.plan.uses_l3
            )
            memo_diagnostic["prompt_has_l1"] = bool(
                context is not None and memory.plan.uses_l1
            )
            memo_diagnostic["memory_tokens_injected"] = memory_tokens
            memo_diagnostic["token_overflow_drops"] = drops
            if isinstance(memo_diagnostic.get("token_budget"), dict):
                memo_diagnostic["token_budget"] = dict(
                    memo_diagnostic["token_budget"]
                )
                memo_diagnostic["token_budget"]["memory_block_tokens"] = memory_tokens
                memo_diagnostic["token_budget"]["token_overflow_drops"] = drops
            self._record_memory_context(result, context, memo_diagnostic)
            l3_diagnostic.update(memo_diagnostic)
            if context is not None:
                surviving = tuple(dict.fromkeys(
                    ref.event_id for item in context.layer_items
                    for ref in item.evidence_refs
                ))
                memory.confirm_context(surviving, step=step)
            else:
                memory.discard_context()
        else:
            task_text, l3_context, l3_diagnostic = self._augment_task_text(
                instruction, step, l3_diagnostic,
            )
            navigation_qs = build_prompt(self.promt_template, task_text)
            verify_task_text_injection(
                self.promt_template, instruction, l3_context, navigation_qs,
            )

        self.last_task_text = task_text
        self.last_prompt_text = navigation_qs
        self.last_prompt_tokens = None
        self.last_memory_context_text = (
            l3_diagnostic.get("memory_context_text")
            if memory is not None else self.last_l3_context_text
        )

        started = time.perf_counter()
        navigation = self.predict_inference(navigation_qs)
        inference_ms = (time.perf_counter() - started) * 1000.0
        self.last_inference_ms = inference_ms
        self.last_prediction = navigation
        
        if self.require_map:
            img = self.addtext(output_im, instruction, navigation)
            self.topdown_map_list.append(img)


        action_list = navigation.split(" ")

        for action in action_list: 
            if action == "stop":
                self.pending_action_list.append(0)
            elif action == "forward":
                self.pending_action_list.append(1)
            elif action == "left":
                self.pending_action_list.append(2)
            elif action == "right":
                self.pending_action_list.append(3)
            else:
                raise ValueError("wrong actions!, please check the code and data")
                                    
            if len(self.pending_action_list) == 2: # to accelrate the inference
                break

        
        selected_action = self.pending_action_list.pop(0)
        self.last_executed_action = {"action": selected_action}
        self._finish_step(
            l3_diagnostic, action=selected_action, inference_ms=inference_ms,
            prompt_tokens=getattr(self, "last_prompt_tokens", None),
            prompt_text=navigation_qs,
        )
        return {"action": selected_action}


def _item_has_l1(item):
    contribution = item.metadata.get("layer_contribution") or {}
    return bool(
        contribution.get("l1")
        or item.metadata.get("layer_source") == "l1"
        or item.metadata.get("role") in {"current", "history"}
    )


def _item_has_l3(item):
    contribution = item.metadata.get("layer_contribution") or {}
    return bool(
        contribution.get("l3")
        or item.metadata.get("layer_source") == "l3"
        or item.metadata.get("object_id") is not None
    )


def _episode_instruction(episode):
    instruction = getattr(episode, "instruction", None)
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    return str(
        getattr(instruction, "instruction_text", None)
        or getattr(instruction, "text", None)
        or ""
    )


def _mean(values):
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _percentile(values, q):
    finite = sorted(float(value) for value in values if value is not None)
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = q * (len(finite) - 1)
    lower = int(position)
    upper = min(lower + 1, len(finite) - 1)
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight
