import numpy as np

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

from l3_integration.experiment_groups import ALL_GROUPS, spec as group_spec
from l3_integration.official_prompt import (
    REFERENCE_OFFICIAL_PROMPT_SHA256,
    build_prompt,
    build_task_text,
    prompt_sha256,
    verify_task_text_injection,
)
from l3_integration.sbadir_fusion import compose_sbadir

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
                 l1_backend=None, l1_max_tokens=160, source_background=False):
        
        print("Initialize UniNaVid")

        if experiment_group in ALL_GROUPS:
            frozen = group_spec(experiment_group)
            if bool(source_background) != frozen.source_background:
                raise ValueError(
                    "source_background does not match frozen group " + experiment_group
                )
            if (l1_backend is not None) != frozen.uses_l1:
                raise ValueError(
                    "L1 backend presence does not match frozen group " + experiment_group
                )
        self.l3 = l3_sidecar
        self.l1 = l1_backend
        self.l1_max_tokens = l1_max_tokens
        #: B enables only the SBAdir evidence/source pairing; C enables it too
        #: and additionally carries the L1 RGB memory.
        self.source_background = bool(source_background)
        self.env = env
        self.config = config
        self.experiment_group = experiment_group

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
        if self.source_background and self.prompt_template_sha256 != REFERENCE_OFFICIAL_PROMPT_SHA256:
            raise RuntimeError("SBAdir requires the unchanged official Uni-NaVid prompt")

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


    def predict_inference(self, prompt):
        

        
        
        
        
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
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
        self.last_prompt_tokens = int(input_ids.shape[1])
        if (getattr(self, "source_background", False)
                and self.last_prompt_tokens + 1024 > self.context_len):
            raise RuntimeError("SBAdir exact input tokens exceed checkpoint context reserve")

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
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
                temperature=0.2,
                max_new_tokens=1024,
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
        self.last_sbadir_memory_text = None
        self.episode_instruction = instruction
        if getattr(self, "l1", None) is not None:
            if episode is None:
                with _preserve_rng_state():
                    self.l1.close_episode()
            else:
                if not self.episode_instruction:
                    self.episode_instruction = _episode_instruction(episode)
                with _preserve_rng_state():
                    self.l1.reset_episode(
                        str(episode.scene_id), str(episode.episode_id),
                        self.episode_instruction,
                    )
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


    def _observe_l1(self, rgb, step):
        backend = getattr(self, "l1", None)
        if backend is None:
            return {"l1_enabled": False}
        try:
            with _preserve_rng_state():
                _events, result = backend.observe(
                    rgb, step, rgb_observation_source="observation.rgb",
                    last_action=self.last_action,
                )
            return {"l1_enabled": True, **{
                "l1_" + str(key): value for key, value in result.items()
            }}
        except Exception as exc:
            # Preserve the navigation path and record the failed RGB channel.
            return {"l1_enabled": True,
                    "l1_observe_error": "%s: %s" % (type(exc).__name__, exc)}


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
            backend = getattr(self, "l1", None)
            link_kwargs = {}
            if backend is not None and getattr(backend, "last_step", None) == step:
                link_kwargs = {
                    "l1_events": backend.last_events,
                    "l1_store": backend.v3_store,
                }
            with _preserve_rng_state():
                observation_diagnostic = sidecar.observe(
                    observations, agent_state, step,
                    transition_record=bool(step > 0),
                    **link_kwargs,
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


    def _augment_sbadir_task_text(self, instruction, step, diagnostic):
        """Pair L3 facts (and optionally L1 facts) with their real sources.

        B passes ``l1=None`` and therefore feeds an empty L1 fact list into the
        same ``compose_sbadir`` code path used by C.  The L3 organizer is read
        once; its facts, order and text are never rewritten.
        """
        backend = getattr(self, "l1", None)
        sidecar = getattr(self, "l3", None)
        if sidecar is None:
            raise RuntimeError("source background requires the real L3 sidecar")
        if backend is not None and not self.source_background:
            raise RuntimeError("an L1 backend requires the source-background mode")
        if not isinstance(instruction, str):
            raise ValueError("instruction must be text")
        with _preserve_rng_state():
            organization, l3_diagnostic = sidecar.build_context(instruction, step)
        if isinstance(l3_diagnostic, dict):
            diagnostic.update(l3_diagnostic)
        l3_context = (
            getattr(getattr(organization, "context", None), "text", None)
            if organization is not None else None
        )
        l3_facts = (
            getattr(organization, "selected_facts", ())
            if organization is not None else ()
        )
        self.last_l3_context_text = l3_context
        l1_facts = ()
        selection = None
        if backend is not None and getattr(backend, "last_step", None) == step:
            with _preserve_rng_state():
                selection = backend.select(step=step, rgb_width=int(self.rgb_list[-1].shape[1]))
            l1_facts = selection.selected_facts
            diagnostic["l1_selection"] = selection.diagnostics
            diagnostic["l1_fact_records"] = [fact.to_dict() for fact in l1_facts]
        elif backend is not None:
            diagnostic["l1_selection_error"] = "L1 observation unavailable at this step"

        def count_tokens_strict(value):
            count = self._count_tokens(value)
            if count is None:
                raise RuntimeError("checkpoint tokenizer did not count prompt tokens")
            return count

        if type(self.context_len) is not int or self.context_len <= 1024 + 64:
            raise RuntimeError("checkpoint context limit cannot hold official generation reserve")
        # The official decoder keeps max_new_tokens=1024.  Reserve 64 tokens
        # for its conversation/visual markers in addition to the text count.
        capacity = self.context_len - 1024 - 64
        fused = compose_sbadir(
            template=self.promt_template, instruction=instruction,
            l3_text=l3_context, l3_facts=l3_facts, l1_facts=l1_facts,
            count_tokens=count_tokens_strict,
            max_prompt_tokens=capacity,
            l1_max_tokens=getattr(self, "l1_max_tokens", 160),
            entity_links=sidecar.cross_sensor_links,
        )
        if fused.diagnostics["omitted_l3_provenance"]:
            raise RuntimeError("L3 source background did not fit measured prompt capacity")
        self.last_sbadir_memory_text = fused.memory_text
        diagnostic.update(fused.diagnostics)
        diagnostic["context_injected"] = bool(fused.memory_text)
        diagnostic["prompt_has_l3"] = bool(l3_context)
        diagnostic["prompt_has_l1"] = bool(fused.l1_text)
        diagnostic["fact_count"] = len(l3_facts)
        diagnostic["context_chars"] = len(l3_context or "")
        # Match A's standalone L3 tokenizer count for A/B reporting.  The
        # marginal full-prompt delta remains separately logged for budgeting.
        diagnostic["context_tokens"] = (
            self._count_tokens(l3_context) if l3_context else 0
        )
        diagnostic["l3_original_memory_unchanged"] = bool(
            fused.diagnostics["l3_core_unchanged"]
        )
        diagnostic["source_background_facts"] = (
            fused.diagnostics["l3_provenance_count"]
            + fused.diagnostics["l1_retained_facts"]
        )
        diagnostic["conflict_type_counts"] = {
            "object_label": fused.diagnostics["conflicting_facts"]
        }
        diagnostic["conflict_evidence_ids"] = [
            {
                "l1": entry["l1_evidence_ids"],
                "l3": entry["l3_evidence_ids"],
            }
            for entry in fused.diagnostics["conflicts"]
        ]
        if selection is not None:
            retained_ids = {
                evidence_id
                for ids in fused.diagnostics["l1_retained_evidence_ids"]
                for evidence_id in ids
            }
            backend.confirm_local_graph(retained_ids, step=step)
        return fused.task_text, fused.memory_text, diagnostic


    @staticmethod
    def _json_default(value):
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        return str(value)


    def _finish_step(self, diagnostic, *, action, inference_ms, prompt_tokens,
                     prompt_text):
        row = dict(diagnostic)
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
            if getattr(self, "source_background", False):
                row["sbadir_memory_text"] = self.last_sbadir_memory_text
        row.setdefault("l3_enabled", getattr(self, "l3", None) is not None)
        row.setdefault("context_injected", False)
        row.setdefault("prompt_has_l3", False)
        row.setdefault("fact_count", 0)
        row.setdefault("context_chars", 0)
        row.setdefault("context_tokens", 0)
        row.setdefault("frame_valid", False)
        row.setdefault("frame_error", None)
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
        self.step_index += 1
        if getattr(self, "source_background", False):
            self.last_action = {"action": action}


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
        map_summary = None
        if sidecar is not None:
            with _preserve_rng_state():
                map_summary = sidecar.map_snapshot()
        env = getattr(self, "env", None)
        scene_id = None
        if env is not None and getattr(env, "current_episode", None) is not None:
            scene_id = getattr(env.current_episode, "scene_id", None)
        summary = {
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
        }
        if getattr(self, "source_background", False):
            backend_present = getattr(self, "l1", None) is not None
            prompt_rows = [row for row in rows if row.get("prompt_tokens") is not None]
            l1_rows = [row for row in rows if row.get("l1_enabled")]
            aggregate_keys = (
                "l1_fact_count", "l1_retained_facts", "l1_omitted_facts",
                "duplicated_facts", "conflicting_facts", "retained_facts",
                "omitted_facts", "source_background_facts",
                "l3_provenance_count", "uncertain_entity_matches",
            )
            totals = {
                key + "_total": sum(int(row.get(key, 0) or 0) for row in prompt_rows)
                for key in aggregate_keys
            }
            summary.update(totals)
            summary.update({
                "source_background_enabled": True,
                "l1_enabled": backend_present,
                "l1_fact_count_total": totals["l1_retained_facts_total"],
                "prompt_has_l1_steps": sum(bool(row.get("prompt_has_l1")) for row in prompt_rows),
                "l1_enabled_steps": sum(bool(row.get("l1_enabled")) for row in rows),
                "l1_observation_attempt_steps": len(l1_rows),
                "l1_observe_error_count": sum(bool(row.get("l1_observe_error")) for row in l1_rows),
                "l1_detector_error_count": sum(bool(row.get("l1_detector_error")) for row in l1_rows),
                "l1_place_attempt_steps": sum(bool(row.get("l1_place_attempted")) for row in l1_rows),
                "l1_place_observed_steps": sum(bool(row.get("l1_place_observed")) for row in l1_rows),
                "l1_place_error_count": sum(bool(row.get("l1_place_error")) for row in l1_rows),
                "l3_segmentation_error_count": sum(bool(row.get("segmentation_error")) for row in l3_rows),
                "cross_label_entity_links_total": sum(
                    int(row.get("cross_label_entity_links", 0) or 0) for row in l3_rows
                ),
                "cross_label_entity_link_error_count": sum(
                    bool(row.get("cross_label_entity_link_error")) for row in l3_rows
                ),
                "l1_context_tokens_mean": _mean([
                    row.get("l1_token_count") for row in prompt_rows
                ]),
                "l3_context_tokens_mean": _mean([
                    row.get("context_tokens") for row in prompt_rows
                    if row.get("prompt_has_l3")
                ]),
                "l3_original_memory_unchanged": bool(prompt_rows) and all(
                    row.get("l3_original_memory_unchanged") is True for row in prompt_rows
                ),
                "conflict_type_counts": {
                    "object_label": sum(int(
                        (row.get("conflict_type_counts") or {}).get("object_label", 0)
                    ) for row in prompt_rows),
                },
                "conflict_evidence_ids": [
                    record for row in prompt_rows
                    for record in row.get("conflict_evidence_ids", [])
                ],
            })
        return summary


    def close(self):
        stream = getattr(self, "_steps_stream", None)
        if stream is not None and not stream.closed:
            stream.flush()
            stream.close()
        sidecar = getattr(self, "l3", None)
        if sidecar is not None:
            with _preserve_rng_state():
                sidecar.close_episode()
        backend = getattr(self, "l1", None)
        if backend is not None:
            with _preserve_rng_state():
                backend.close_episode()


    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)
        instruction = observations["instruction"]["text"]
        step = self.step_index
        l1_diagnostic = self._observe_l1(rgb, step)
        l3_diagnostic = self._observe_l3(observations, step)
        l3_diagnostic.update(l1_diagnostic)

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, instruction, "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            self._finish_step(
                l3_diagnostic, action=temp_action, inference_ms=None,
                prompt_tokens=None, prompt_text=None,
            )
            return {"action": temp_action}


        if getattr(self, "source_background", False):
            task_text, l3_context, l3_diagnostic = self._augment_sbadir_task_text(
                instruction, step, l3_diagnostic,
            )
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

        started = time.perf_counter()
        navigation = self.predict_inference(navigation_qs)
        inference_ms = (time.perf_counter() - started) * 1000.0
        self.last_inference_ms = inference_ms
        
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
        self._finish_step(
            l3_diagnostic, action=selected_action, inference_ms=inference_ms,
            prompt_tokens=getattr(self, "last_prompt_tokens", None),
            prompt_text=navigation_qs,
        )
        return {"action": selected_action}


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
