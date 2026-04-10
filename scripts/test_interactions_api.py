"""Quick prototype to test the Interactions API with function calling + images."""

import base64
import io
import json

from google import genai
from PIL import Image

# Create a simple test image
def make_test_image(color: str = "red", size: tuple = (100, 100)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def main():
    from langslice.vlm_config import get_client
    client = get_client()

    # Define a simple tool
    view_tool = {
        "type": "function",
        "name": "view_image",
        "description": "Returns a test image for inspection.",
        "parameters": {
            "type": "object",
            "properties": {
                "color": {"type": "string", "description": "Color of the test image"},
            },
            "required": ["color"],
        },
    }

    place_tool = {
        "type": "function",
        "name": "place_point",
        "description": "Places a point at the given coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"},
            },
            "required": ["x", "y"],
        },
    }

    finish_tool = {
        "type": "function",
        "name": "finish",
        "description": "Complete the task.",
        "parameters": {"type": "object", "properties": {}},
    }

    tools = [view_tool, place_tool, finish_tool]

    # Turn 1: Initial request
    print("=== Turn 1: Initial request ===")
    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="You have tools to view images and place points. Start by calling view_image with color 'blue'. Then place a point at (50, 50). Then finish.",
        tools=tools,
        system_instruction="You are a test assistant. Follow instructions exactly.",
    )
    print(f"  ID: {interaction.id}")
    print(f"  Status: {interaction.status}")
    print(f"  Outputs: {len(interaction.outputs)}")
    for i, output in enumerate(interaction.outputs):
        print(f"  Output[{i}]: type={output.type}, text={getattr(output, 'text', None)}")
        if hasattr(output, 'name'):
            print(f"    Function call: {output.name}({getattr(output, 'arguments', {})})")
        # Print all attributes
        for attr in dir(output):
            if not attr.startswith('_'):
                val = getattr(output, attr, None)
                if val is not None and not callable(val):
                    print(f"    .{attr} = {repr(val)[:200]}")

    # Check if there's a function call we need to respond to
    # Look for function_call outputs
    fc_output = None
    for output in interaction.outputs:
        if getattr(output, 'type', None) == 'function_call':
            fc_output = output
            break

    if not fc_output:
        print("  No function call found. Dumping raw interaction:")
        print(f"  {interaction}")
        return

    call_id = getattr(fc_output, 'id', None) or getattr(fc_output, 'call_id', None)
    print(f"\n  Function call: {fc_output.name}, call_id={call_id}")

    # Turn 2: Send function result with image
    print("\n=== Turn 2: Function result with image ===")
    test_image = make_test_image("blue")
    b64_image = base64.b64encode(test_image).decode("utf-8")

    function_result = {
        "type": "function_result",
        "call_id": call_id,
        "name": fc_output.name,
        "result": {
            "items": [
                {"type": "text", "text": "Here is your blue test image:"},
                {"type": "image", "data": b64_image, "mime_type": "image/jpeg"},
            ]
        },
    }

    interaction2 = client.interactions.create(
        model="gemini-3-flash-preview",
        input=[function_result],
        tools=tools,
        previous_interaction_id=interaction.id,
        system_instruction="You are a test assistant. Follow instructions exactly.",
    )
    print(f"  ID: {interaction2.id}")
    print(f"  Status: {interaction2.status}")
    print(f"  Outputs: {len(interaction2.outputs)}")
    for i, output in enumerate(interaction2.outputs):
        print(f"  Output[{i}]:")
        for attr in dir(output):
            if not attr.startswith('_'):
                val = getattr(output, attr, None)
                if val is not None and not callable(val):
                    print(f"    .{attr} = {repr(val)[:200]}")

    print("\n=== Success! Interactions API works with function calling + images ===")


if __name__ == "__main__":
    main()
