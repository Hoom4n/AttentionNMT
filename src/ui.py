import os
import gradio as gr

def read_file(path: str, default_content: str = "") -> str:
    """
    Ensure file exists (with default_content if missing) and return its contents.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default_content)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
        
def build_demo(
    generation_fn,
    inputs,
    outputs,
    english_title: str,
    persian_title: str,
    assets_dir: str = "assets",
    app_title: str = "Demo"
):
    """
    args:
        generation_fn: callable for inference
        inputs: list of Gradio input components
        outputs: list of Gradio output components
    """
    md_dir = os.path.join(assets_dir, "markdown")
    css_dir = os.path.join(assets_dir, "css")
    english_md = os.path.join(md_dir, "english_summary.md")
    persian_md = os.path.join(md_dir, "persian_summary.md")
    english_summary = read_file(english_md)
    persian_summary = read_file(persian_md)

    css_file = os.path.join(css_dir, "custom.css")
    css = read_file(css_file, "/* Custom CSS overrides */\n")

    with gr.Blocks(css=css, title=app_title) as demo:
        title_md = gr.Markdown(english_title, elem_id="title")

        with gr.Row():
            english_btn = gr.Button("English")
            persian_btn = gr.Button("فارسی (Persian)")

        summary_md = gr.Markdown(english_summary, elem_id="summary")

        # generation panel
        with gr.Row(variant="panel"):
            with gr.Column(scale=1, variant="panel"):
                for inp in inputs:
                    inp.render()
                generate_btn = gr.Button("✨ Translate", variant="primary")

            with gr.Column(scale=1, variant="panel"):
                for out in outputs:
                    out.render()

        # events
        generate_btn.click(generation_fn, inputs=inputs, outputs=outputs)

        def set_english():
            return (
                gr.update(value=english_title, elem_classes=[]),
                gr.update(value=english_summary, elem_classes=[]),
            )

        def set_persian():
            return (
                gr.update(value=persian_title, elem_classes=["persian"]),
                gr.update(value=persian_summary, elem_classes=["persian"]),
            )

        english_btn.click(set_english, outputs=[title_md, summary_md])
        persian_btn.click(set_persian, outputs=[title_md, summary_md])

    return demo
