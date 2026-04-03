"""Quick test: replay AI Studio's second pass with the exact same images."""

import os
from google import genai
from google.genai import types
from PIL import Image
import io


def main():
    from langslice.ai.config import get_client
    client = get_client()

    # Load the exact images from the AI Studio session
    atlas_annotated = Image.open("docs/first_pass_ai_studio_atlas_image.jpg")
    slice_raw = Image.open("docs/ai_studio_slice_image.png")

    print(f"Atlas annotated: {atlas_annotated.size} ({atlas_annotated.mode})")
    print(f"Slice raw: {slice_raw.size} ({slice_raw.mode})")

    def img_to_part(img, fmt="PNG", mime="image/png"):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format=fmt)
        return types.Part.from_bytes(mime_type=mime, data=buf.getvalue())

    # Use the raw JPEG bytes for the atlas (same as AI Studio)
    with open("docs/first_pass_ai_studio_atlas_image.jpg", "rb") as f:
        atlas_bytes = f.read()
    atlas_part = types.Part.from_bytes(mime_type="image/jpeg", data=atlas_bytes)

    prompt = (
        "This is a mouse brain slice and a corresponding anatomical atlas. "
        "The anatomical atlas is annotated with landmark points. "
        "I'd like you to place corresponding point annotations on the "
        "histological slice in the same relative anatomical positions as "
        "the atlas points. Ensure the numbers of the annotations are preserved.\n\n"
        "Output the histology slice as an image edited with the point "
        "annotations. Please ensure the points are small, such that they "
        "do not block local features."
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                img_to_part(slice_raw),   # slice first (PNG)
                atlas_part,                # annotated atlas second (JPEG)
                types.Part.from_text(text=prompt),
            ],
        )
    ]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        image_config=types.ImageConfig(image_size="1K"),
        response_modalities=["IMAGE", "TEXT"],
    )

    print("\nCalling Gemini...")
    os.makedirs("test_runs/ai_studio_replay", exist_ok=True)

    for chunk in client.models.generate_content_stream(
        model="gemini-3.1-flash-image-preview",
        contents=contents,
        config=config,
    ):
        if chunk.parts is None:
            continue
        for part in chunk.parts:
            if part.inline_data and part.inline_data.data:
                img = Image.open(io.BytesIO(part.inline_data.data))
                img.save("test_runs/ai_studio_replay/slice_annotated.png")
                print(f"  Saved image: {img.size}")
            elif part.text:
                snippet = part.text[:120].replace("\n", " ")
                print(f"  Text: {snippet}")

    print("\nDone. Check test_runs/ai_studio_replay/slice_annotated.png")


if __name__ == "__main__":
    main()
