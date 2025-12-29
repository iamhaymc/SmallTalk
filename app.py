import os, sys
from pathlib import Path

__dir_path__ = Path(os.path.realpath(__file__)).parent
sys.path.append(str(__dir_path__))

import gradio as gr
import spaces

from llm import (
    get_chat_model,
    run_chat_model,
    free_chat_model,
    DEFAULT_CHAT_PRELUDE,
    DEFAULT_MODEL_CHAT_XS_LOCAL,
    DEFAULT_MODEL_CHAT_XS_REMOTE,
    DEFAULT_MODEL_CHAT_XL_LOCAL,
    DEFAULT_MODEL_CHAT_XL_REMOTE,
    get_text_classifier,
    run_text_classifier,
    free_text_classifier,
    DEFAULT_MODEL_TEXT_CLASSIFIER_LOCAL,
    get_text_solver,
    run_text_solver,
    free_text_solver,
    DEFAULT_MODEL_TEXT_SOLVER_LOCAL,
    get_seed,
    SEED_MIN,
    SEED_MAX,
)

CHAT_MODEL_OPTIONS = [
    DEFAULT_MODEL_CHAT_XS_LOCAL,
    DEFAULT_MODEL_CHAT_XS_REMOTE,
    DEFAULT_MODEL_CHAT_XL_LOCAL,
    DEFAULT_MODEL_CHAT_XL_REMOTE,
]

# =================================================================================================

_hf_user = os.getenv("HF_USER", None)
if not _hf_user:
    raise ValueError("HF_USER environment variable is not set.")

_hf_token = os.getenv("HF_TOKEN", None)
if not _hf_token:
    raise ValueError("HF_TOKEN environment variable is not set.")

_data_dir = __dir_path__.joinpath("data")
_data_dir.mkdir(parents=True, exist_ok=True)

_model_dir = __dir_path__.joinpath("models")
_model_dir.mkdir(parents=True, exist_ok=True)

_chat_choice, _chat_model, _chat_tokzr = None, None, None

# _txtcls_pipe = get_text_classifier(model_path=DEFAULT_MODEL_TEXT_CLASSIFIER_LOCAL)

# _txtqa_model, _txtqa_tokzr = get_text_solver(model_path=DEFAULT_MODEL_TEXT_SOLVER_LOCAL)

# =================================================================================================


@spaces.GPU(duration=120)
def generate_response(
    inputs,
    history,
    prelude,
    max_tokens,
    max_seconds,
    seed,
    temperature,
    beams,
    top_k,
    top_p,
    min_p,
    repenalty,
    chat_choice,
):
    global _chat_choice
    global _chat_model
    global _chat_tokzr

    # Ensure model
    if chat_choice != _chat_choice or _chat_model is None:
        if _chat_model is not None:
            free_chat_model(_chat_model)
        use_local = chat_choice.startswith("./")
        _chat_path = _model_dir.joinpath(chat_choice) if use_local else chat_choice
        _chat_model, _chat_tokzr = get_chat_model(_chat_path, use_local=use_local)
        _chat_choice = chat_choice

    # Invoke model
    outputs = []
    for token in run_chat_model(
        model=_chat_model,
        tokzr=_chat_tokzr,
        message=inputs,
        prelude=prelude,
        history=history,
        max_tokens=max_tokens,
        max_time=max_seconds if max_seconds > 0 else None,
        seed=seed,
        temperature=temperature,
        beams=beams,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repenalty=repenalty,
        stream=True,
    ):
        outputs.append(token)
        yield "".join(outputs)


# =================================================================================================


def launch_gradio():
    demo = gr.ChatInterface(
        fn=generate_response,
        type="messages",
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            label="Input",
            file_types=["image", "video"],
            file_count="multiple",
            placeholder="Enter text or upload media",
        ),
        stop_btn="Stop",
        additional_inputs=[
            gr.Textbox(label="Prelude", lines=1, value=DEFAULT_CHAT_PRELUDE),
            gr.Slider(label="Max Tokens", minimum=100, maximum=2000, step=1, value=512),
            gr.Slider(label="Max Seconds", minimum=-1, maximum=60, step=1, value=0),
            gr.Slider(
                label="Seed",
                minimum=SEED_MIN,
                maximum=SEED_MAX,
                step=1,
                value=get_seed(),
            ),
            gr.Slider(
                label="Temperature", minimum=0.1, maximum=2.0, step=0.1, value=0.7
            ),
            gr.Slider(label="Beams", minimum=0, maximum=10, step=1, value=1),
            gr.Slider(label="Top-k", minimum=1, maximum=100, step=1, value=20),
            gr.Slider(label="Top-p", minimum=0.05, maximum=1.0, step=0.05, value=0.8),
            gr.Slider(label="Min-p", minimum=0.0, maximum=1.0, step=0.05, value=0.05),
            gr.Slider(
                label="Repenalty", minimum=1.0, maximum=2.0, step=0.05, value=1.0
            ),
            gr.Dropdown(
                label="Model", value=CHAT_MODEL_OPTIONS[0], choices=CHAT_MODEL_OPTIONS
            ),
        ],
        cache_examples=False,
        fill_width=False,
        fill_height=True,
        theme=gr.themes.Default(),
    )
    demo.launch()


if __name__ == "__main__":
    launch_gradio()
