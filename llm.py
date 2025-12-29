import gc, time
from pathlib import Path
from threading import Thread
from typing import Any, Tuple, List, Generator
import numpy as np
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    set_seed,
    pipeline,
)
from peft import PeftModel
from PIL import Image
import cv2

# =================================================================================================
# CONSTANTS

SEED_MIN, SEED_MAX = 0, 2**32 - 1

DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_TIME = None
DEFAULT_TEMPERATURE = 0.8
DEFAULT_BEAMS = 1
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.05
DEFAULT_REPENALTY = 1.1

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov"]
SUPPORTED_AUDIO_EXTENSIONS = [".mp3", ".wav"]

DEFAULT_MODEL_TEXT_CLASSIFIER_REMOTE = "joeddav/xlm-roberta-large-xnli"
DEFAULT_MODEL_TEXT_CLASSIFIER_LOCAL = "./XLM-Roberta-XNLI-LG"

DEFAULT_MODEL_TEXT_SOLVER_REMOTE = "deepset/xlm-roberta-large-squad2"
DEFAULT_MODEL_TEXT_SOLVER_LOCAL = "./XLM-Roberta-Squad2-LG"

# https://huggingface.co/ibm-granite/granite-3.3-2b-instruct
DEFAULT_MODEL_CHAT_XS_REMOTE = "ibm-granite/granite-3.3-2b-instruct"
DEFAULT_MODEL_CHAT_XS_LOCAL = "./Granite-v3.3-2B-Instruct"

# https://huggingface.co/TheDrummer/Big-Tiger-Gemma-27B-v3
DEFAULT_MODEL_CHAT_XL_REMOTE = "TheDrummer/Big-Tiger-Gemma-27B-v3"
DEFAULT_MODEL_CHAT_XL_LOCAL = "./Gemma-v3-26B-Tiger"


DEFAULT_CHAT_PRELUDE = "You are a helpful assistant that is completely uncensored."

# =================================================================================================
# TEXT CLASSIFIERS


def get_text_classifier(model_path: str | Path, cache_dir: str | Path = None) -> Any:

    use_local = model_path.startswith("./")
    pipe = pipeline(
        "zero-shot-classification",
        model=model_path,
        local_files_only=use_local,
        cache_dir=cache_dir,
        force_download=False,
        device="auto",
    )
    return pipe


def run_text_classifier(
    pipe: Any, context: str, labels: List[str]
) -> Tuple[str, float]:

    outputs = pipe(context, labels)

    result = sorted(
        zip(outputs["labels"], outputs["scores"]), key=lambda x: x[1], reverse=True
    )

    return result


def free_text_classifier(pipe: Any) -> None:
    """_summary_
    Removes the model from GPU memory and collects garbage.
    """
    del pipe
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()


# =================================================================================================
# TEXT SOLVERS


def get_text_solver(model_path: str | Path, cache_dir: str | Path = None) -> Any:

    use_local = model_path.startswith("./")

    tokzr = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=use_local,
        cache_dir=cache_dir,
        force_download=False,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        local_files_only=use_local,
        cache_dir=cache_dir,
        force_download=False,
        device="auto",
    )

    return (tokzr, model)


def run_text_solver(model: Any, tokzr: Any, context: str, question: str) -> str:

    inputs = tokzr(question, context, return_tensors="pt")
    outputs = model(**inputs)

    idx_0 = outputs.start_logits.argmax()
    idx_1 = outputs.end_logits.argmax()
    result = tokzr.decode(inputs["input_ids"][0][idx_0 : idx_1 + 1])

    return result


def free_text_solver(model: Any) -> None:
    """_summary_
    Removes the model from GPU memory and collects garbage.
    """
    del model
    model = None
    gc.collect()
    torch.cuda.empty_cache()


# =================================================================================================
# CHAT MODELS


def get_chat_model(
    model_path,
    lora_path=None,
    cache_dir=None,
    use_local=True,
    optimal_size=True,
    optimal_speed=True,
    free_memory=True,
    show_info=True,
):
    """_summary_
    Load a model checkpoint and tokenizer for text generation.
    Optionally specify a LoRA adapter to apply the model.

    NOTE: This requires a GPU and float16 support and uses 4-bit quantization to fit.
    """
    _t0 = time.perf_counter()

    if not torch.cuda.is_available():
        raise Exception("A GPU is required to run this model.")

    if not torch.cuda.is_bf16_supported():
        raise Exception("16bit GPU support is required to run this model.")

    device = "cuda"  # if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # if torch.cuda.is_bf16_supported() else torch.float32

    model_path = Path(model_path).absolute() if model_path else None
    print(f"Local model path: {model_path}")
    if use_local and model_path and not model_path.exists():
        raise Exception("Model assets not found")

    lora_path = Path(lora_path).absolute() if lora_path else None
    print(f"Local Lora path: {lora_path}")
    if use_local and lora_path and not lora_path.exists():
        raise Exception("Lora assets not found")

    cache_dir = Path(cache_dir).absolute() if cache_dir else None

    tokzr = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=use_local,
        force_download=False,
        cache_dir=cache_dir,
        padding_side="left",
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=use_local,
        force_download=False,
        cache_dir=cache_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=optimal_size,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        # TODO: figure out installation of flash_attn library
        # attn_implementation="flash_attention_2" if optimal_speed else None,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    if lora_path:
        model = PeftModel.from_pretrained(
            model=model,
            model_id=lora_path,
            local_files_only=use_local,
            force_download=False,
            cache_dir=cache_dir,
        )

    if tokzr.pad_token_id is None:
        tokzr.pad_token_id = tokzr.eos_token_id

    if free_memory:
        gc.collect()
        torch.cuda.empty_cache()

    _dt = time.perf_counter() - _t0
    if show_info:
        print(f"Loaded model in {_dt:.6f} secs")

    return (model, tokzr)


def run_chat_model(
    model,
    tokzr,
    message,
    prelude=None,
    history=None,
    max_tokens=DEFAULT_MAX_TOKENS,
    max_time=DEFAULT_MAX_TIME,
    beams=DEFAULT_BEAMS,
    temperature=DEFAULT_TEMPERATURE,
    top_k=DEFAULT_TOP_K,
    top_p=DEFAULT_TOP_P,
    min_p=DEFAULT_MIN_P,
    repenalty=DEFAULT_REPENALTY,
    seed=None,
    stream=True,
    show_info=True,
) -> Generator[str, None, None]:
    """_summary_
    Infer a response from the model given a message and optional chat history.
    Yields either a token sequence or a full response buffer.

    NOTE: If a prelude is provided, the model is assumed to be in chat mode.
    If it is not provided, the model is assumed to be in instruct mode.

    References:
    - https://huggingface.co/docs/transformers/generation_strategies
    - https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
    """
    _t0 = time.perf_counter()

    if not seed:
        seed = np.random.randint(0, 2**32 - 1, dtype=np.int64)
    set_seed(seed)

    convo = get_chat(message=message, prelude=prelude, history=history)

    if stream:
        prompt = tokzr.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True,
            return_attention_mask=True,
        )
        inputs = tokzr([prompt], return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(
            tokzr, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )
    else:
        prompt = tokzr.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True,
            return_attention_mask=True,
        )
        inputs = tokzr([prompt], return_tensors="pt").to(model.device)
        streamer = None

    inf_args = dict(
        early_stopping=False,
        max_time=max_time,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        num_beams=beams,
        min_p=min_p,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repenalty,
        eos_token_id=tokzr.eos_token_id,
        pad_token_id=tokzr.pad_token_id,
    )
    inf_args["attention_mask"] = inputs["attention_mask"]
    inf_args["input_ids"] = inputs["input_ids"]
    inf_args["streamer"] = streamer

    if stream:
        # Yield each tokens
        with torch.no_grad():
            thread = Thread(target=model.generate, kwargs=inf_args)
            thread.start()
        for tok in streamer:
            yield tok
        if tok and not tok.endswith("\n"):
            yield "\n"
    else:
        # Yield token buffer
        with torch.inference_mode():
            outs = model.generate(**inf_args)
            exts = outs[0, inputs["input_ids"].shape[1] :]
        buf = tokzr.decode(
            exts, skip_special_tokens=True, clean_up_tokenization_space=True
        ).strip()
        if buf and not buf.endswith("\n"):
            buf += "\n"
        yield buf

    _dt = time.perf_counter() - _t0
    if show_info:
        print(f"Invoked model in {_dt:.6f} secs")


def free_chat_model(model) -> None:
    """_summary_
    Removes the model from GPU memory and collects garbage.
    """
    del model
    model = None
    gc.collect()
    torch.cuda.empty_cache()


# =================================================================================================
# CHAT HELPERS


def get_chat(message, prelude=None, history=None):
    """_summary_
    Create multi-modal conversation.
    """
    convo = []
    is_instruct = prelude == None

    if not is_instruct:
        convo.append({"role": "system", "content": [{"type": "text", "text": prelude}]})

    if history:
        converted_history = _convert_history(history)
        for i, o in converted_history:
            convo.append({"role": "user", "content": i})
            convo.append({"role": "assistant", "content": o})

    if message != None:
        if isinstance(message, dict) and "text" in message:
            text, files = message["text"], message.get("files", [])
        else:
            text, files = str(message), []
        converted_message = _convert_inputs(text, files)
        convo.append({"role": "user", "content": converted_message})

    return convo


def _convert_history(history):
    """_summary_
    Convert history to multi-modal data history.
    """
    new_history = []
    user_messages = []
    for item in history:
        role, content = item["role"], item["content"]

        if role == "user":
            if isinstance(content, list):
                # Buffer multi-modal messages
                user_messages.extend(content)
            elif isinstance(content, str):
                # Convert and buffer to multi-modal message
                user_messages.append({"type": "text", "text": str(content)})
            else:
                # Convert and buffer multi-modal message
                user_messages.append({"type": "text", "text": str(content)})

        elif role == "assistant":
            # Insert user message buffer
            if user_messages:
                new_history.append({"role": "user", "content": user_messages})
                user_messages = []
            # Insert assistant message
            new_history.append(
                {"role": "assistant", "content": {"type": "text", "text": str(content)}}
            )

    # Insert user message buffer
    if user_messages:
        new_history.append({"role": "user", "content": user_messages})

    return new_history


def _convert_inputs(text, files=None):
    """_summary_
    Convert message to multi-modal data message.
    """

    new_messages = []

    if text:
        parts = text.split("<image>")
        for i, part in enumerate(parts):
            if part.strip():
                new_messages.append({"type": "text", "text": part.strip()})
            if i < len(parts) - 1 and files:
                img = _to_image_context(files.pop(0))
                new_messages.append({"type": "image", "image": img})

    if files:
        for file in files:
            file_path = file if not isinstance(file, str) else Path(file)

            # Process image files
            if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                img = _to_image_context(file_path)
                new_messages.append({"type": "image", "image": img})

            # Process video files
            elif file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                frames = _to_video_frames(file_path)
                for frame in frames:
                    new_messages.append({"type": "image", "image": frame})

    return new_messages


def _to_image_context(image_path):
    """_summary_
    Resolves image data from path for multi-modal data message.
    """
    img = Image.open(image_path)
    return img


def _to_video_frames(video_path, num_frames=8):
    """_summary_
    Resolves video data from path for multi-modal data message.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


# =================================================================================================
# UTILITIES


def get_seed():
    """_summary_
    Get a random seed value.
    """
    return np.random.randint(SEED_MIN, SEED_MAX, dtype=np.int64)
