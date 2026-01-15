import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
import gradio as gr

print("Loading GLM-Image model...")

# A README-ben is így töltik be a modellt
# https://huggingface.co/zai-org/GLM-Image és a GitHub README alapján :contentReference[oaicite:1]{index=1}
pipe = GlmImagePipeline.from_pretrained(
    "zai-org/GLM-Image",
    torch_dtype=torch.bfloat16,
    device_map="cuda",  # itt NEM "auto", hanem "cuda" kell
)


def generate_image(prompt, height, width, steps, guidance, seed):
    if not prompt or not prompt.strip():
        return None

    # seed kezelése
    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        except ValueError:
            # ha valami hülyeség kerül be, ignoráljuk és random seed lesz
            generator = None

    # GLM-Image hivatalos hívási minta (README)
    # image = pipe(
    #     prompt=prompt,
    #     height=32 * 32,
    #     width=36 * 32,
    #     num_inference_steps=50,
    #     guidance_scale=1.5,
    #     generator=torch.Generator(device="cuda").manual_seed(42),
    # ).images[0] :contentReference[oaicite:2]{index=2}

    out = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    )

    return out.images[0]


with gr.Blocks() as demo:
    gr.Markdown("## GLM-Image képgenerálás (RunPod / Gradio)")

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Pl.: A beautifully designed modern food magazine style dessert recipe illustration...",
            lines=4,
        )

    with gr.Row():
        # GLM-Image: a cél felbontásnak 32-vel oszthatónak kell lennie :contentReference[oaicite:3]{index=3}
        height = gr.Slider(
            512, 2048,
            value=1024,
            step=32,
            label="Magasság (height, 32-vel osztható)"
        )
        width = gr.Slider(
            512, 2048,
            value=1024,
            step=32,
            label="Szélesség (width, 32-vel osztható)"
        )

    with gr.Row():
        steps = gr.Slider(
            10, 100,
            value=50,
            step=1,
            label="Lépésszám (num_inference_steps)"
        )
        guidance = gr.Slider(
            0.5, 3.0,
            value=1.5,   # gyári ajánlott érték :contentReference[oaicite:4]{index=4}
            step=0.1,
            label="Guidance scale (CFG-szerű)"
        )

    seed = gr.Number(label="Seed (opcionális, int)", value=42)

    generate_btn = gr.Button("Generálás")
    output_image = gr.Image(label="Generált kép",format="png")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, steps, guidance, seed],
        outputs=output_image,
    )

# FONTOS: RunPod miatt 0.0.0.0 és fix port 8890
demo.launch(server_name="0.0.0.0", server_port=8890, share=True)
