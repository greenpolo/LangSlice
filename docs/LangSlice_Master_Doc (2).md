# **Vision**:

I’m building an AI agent which aligns 2D brain slice images to neuroanatomical atlases available in the BrainGlobe Atlas API. 

The agent will follow a basic 2-part workflow which the user will be able to independently initiate. The first will involve the agent exploring the atlases Z-axis, anterior to posterior, to find the position along the atlas which the selected slice belongs to. The second part will involve the agent (in a separate call/runtime) placing paired coordinate points on both the atlas and slice, such that an algorithm could apply XY translations and thin-plate spline warps to register the atlas overlapping onto the histology slice. 

**Workflow**:

**AP Estimation**

**AP Estimation Workflow for Text-Centric Multimodal Models:**  
(gemini-3.1-pro, gemini-3-flash, gemini-3.1-flash-lite)

For the text-centric LLMs, we’ll use a tools-based approach for the AP/Bregma estimation. We’ll start by giving the agent detailed step-by instructions on how to most effectively conduct the estimation, then give it freedom. We’ll provide the model with many custom tools it can use to find the correct position, then let it autonomously reason its way to the correct position.

"You are an expert neuroanatomist. You are given a histology brain slice image and must determine its Anterior-Posterior (AP) position within a reference atlas. The coordinate system is: 0.0 mm is the extreme anterior edge (e.g. olfactory bulb), while larger mm values move posterior toward the cerebellum and brainstem. You have tools to fetch atlas reference images at any AP coordinate, query which brain regions exist at a given position, and get atlas metadata”

“RECOMMENDED STRATEGY

1\. Coarse Sweep: Call \`fetch\_multiple\_atlas\_slices\` with 4-5 widely spaced coordinates (e.g., 2.0, 4.0, 6.0, 8.0) as your first real image search step to instantly find the correct neighborhood

2\. Finer Search: Identify the closest match, then call \`fetch\_multiple\_atlas\_slices’ again around that match with tighter spacing (e.g., \+/-0.5 mm)

3\. Verification: Once narrowed down, check specific structural landmarks or use \`get\_region\_names\` to confirm anatomical identity. Before submitting, compare at least one lower and one higher neighboring AP position around your leading candidate.

4\. Submit: Call \`submit\_estimate\` only when you are highly confident. Do not guess blindly; use the tools to narrow down the answer methodically. Avoid long thought-only turns: either perform the next search step or submit once the neighborhood is bracketed."

**AP Estimation Workflow for Image-gen Models**  
(gemini-3-pro-image, gemini-3.1-flash-image) 

Since the image-gen models can't pick tools or call functions, we can create a workflow that spoon-feeds the decision-making process to the model. The models can accept up to 14 images in one prompt as input. The basic idea being, on each call, increase the zoom of the atlas, up to 0.05mm spacing between atlas images. Once we’ve reached 0.05mm resolution, we stop.

1\. First Call   
This is a \[insert animal\] histological brain section, and \[10-13\] images of a corresponding neuro-anatomical atlas (images are evenly spaced throughout the atlas). Please indicate which of these atlas images is most similar to the histological slice \- it won’t be a perfect match. Please only output text.

2\. Second Call   
This is a \[insert animal\] histological brain section, and \[10-13\] images of a corresponding neuro-anatomical atlas (zoomed in on the neighborhood of the first answer). Please indicate which of these atlas images is most similar to the histological slice \- it won’t be a perfect match. Please only output text.

3\. Third Call   
This is a \[insert animal\] histological brain section, and \[10-13\] images of a corresponding neuro-anatomical atlas (zoomed in to a resolution of 0.05mm distance per image). Please indicate which of these atlas images is most similar to the histological slice \- it won’t be a perfect match. Please only output text.

**Landmark Point Placement**

After lots of testing, I’ve found that vision generation models, specifically gemini-3-pro-image (nano banana pro) and gemini-3.1-flash-image (nano banana 2\) are very good at processing many landmarks at a single time with high accuracy, however, these models are unable to call functions/tools.On the other hand, multimodal LLMs like gemini-3.1-pro and gemini-3-flash struggle with placing many paired coordinates in a single turn, but they benefit from the ability to call functions/tools.

Identified two optimal workflows, one for the text-based multimodal LLMs, the other is for the image generation models. The text-based workflow provides the model with a small selection of tools it can use to inspect the image, place visible point annotations (that are encoded such that they appear on the image for the model to see in subsequent tool calls), view the atlas and slice points side-by-side at varying magnifications to verify point placement, adjust the point(s) if needed, then save the point(s). The image-based workflow accounts for the image-gen model’s lack of agentic ability via a 2-pass system: first, we call the first image gen model to generate point annotations on the atlas image in a single turn, then we send a prompt which includes the newly annotated atlas alongside the histology slice image to another image gen model telling it to transfer the atlas point annotations to the relative anatomy of the histology slice.

**New Landmark Workflow for Image-Gen Models:**  
(gemini-3-pro-image, gemini-3.1-flash-image) 

1\. First Call  
This is a coronal section of a \[insert animal\] brain anatomical atlas. I’d like you to place landmark points as annotations on the atlas. Prioritize placing landmarks which are evenly distributed, visually distinct, and would be easily identifiable in a real brain section or easily identifiable if described in text. Please place \[n\] points on the outline/border of the brain slice atlas, and \[n\] points in the interior of the brain slice atlas. Label each annotation with a number.

Output the atlas as an image edited with the point annotations.Please ensure the points are small, such that they do not block local features. 

2\. Second Call  
This is an \[insert animal\] brain slice and a corresponding anatomical atlas. The anatomical atlas is annotated with landmark points. I'd like you to place corresponding point annotations on the histological slice in the same relative anatomical positions as the atlas points. Ensure the numbers of the annotations are preserved. 

Output the histology slice as an image edited with the point annotations. Please ensure the points are small.

Code:

\# To run this code you need to install the following dependencies:  
\# pip install google-genai

import mimetypes  
import os  
from google import genai  
from google.genai import types

def save\_binary\_file(file\_name, data):  
    f \= open(file\_name, "wb")  
    f.write(data)  
    f.close()  
    print(f"File saved to to: {file\_name}")

def generate():  
    client \= genai.Client(  
        api\_key=os.environ.get("GEMINI\_API\_KEY"),  
    )

    model \= "gemini-3.1-flash-image-preview"  
    contents \= \[  
        types.Content(  
            role="user",  
            parts=\[  
                types.Part.from\_text(text="""INSERT\_INPUT\_HERE"""),  
            \],  
        ),  
    \]  
    generate\_content\_config \= types.GenerateContentConfig(  
        thinking\_config=types.ThinkingConfig(  
            thinking\_level="MINIMAL",  
        ),  
        image\_config \= types.ImageConfig(  
            aspect\_ratio="",  
            image\_size="1K",  
            person\_generation="",  
        ),  
        response\_modalities=\[  
            "IMAGE",  
        \],  
    )

    file\_index \= 0  
    for chunk in client.models.generate\_content\_stream(  
        model=model,  
        contents=contents,  
        config=generate\_content\_config,  
    ):  
        if (  
            chunk.parts is None  
        ):  
            continue  
        if chunk.parts\[0\].inline\_data and chunk.parts\[0\].inline\_data.data:  
            file\_name \= f"ENTER\_FILE\_NAME\_{file\_index}"  
            file\_index \+= 1  
            inline\_data \= chunk.parts\[0\].inline\_data  
            data\_buffer \= inline\_data.data  
            file\_extension \= mimetypes.guess\_extension(inline\_data.mime\_type)  
            save\_binary\_file(f"{file\_name}{file\_extension}", data\_buffer)  
        else:  
            print(chunk.text)

if \_\_name\_\_ \== "\_\_main\_\_":  
    generate()

**Landmark Workflow for Text-Centric Multimodal Models:**  
(gemini-3.1-pro, gemini-3-flash, gemini-3.1-flash-lite)

I want you to place matched landmark points on the slice and atlas images.

Work on one point at a time, or at most a very small batch of closely related points. Do not place many points at once.

For each point, follow this workflow:

1\. Define the target feature before placing anything.  
   \- First state the exact local anatomical/geometric feature you are trying to match in the atlas and in the slice.  
   \- Use rich, specific descriptions such as “the deepest point of the dorsal midline notch” or “the lower third of the intact medial wall of the right lateral ventricle,” not broad labels like “in the ventricle” or “on the border.”

2\. Place the initial points as visible annotations.  
   \-Use the place annotation tool, in standardized coordinates.  
   \-Place the points in both the atlas and the slice image.  
   \- Choose the same kind of local feature in both images.  
   \- Prioritize local correspondence in depth, curvature, neighboring contours, and boundary context over global shape similarity.

3\. Zoom in (3x) and inspect locally.  
   \- Zoom in and compare the atlas and slice side-by-side.  
   \- Adjust the point positions if needed

4\. Then zoom out (1.5x) and sanity-check in a broader context.  
   \- After local matching, zoom out and verify that the point still makes sense within the broader anatomy.  
   \- Use broader anatomical context only as a validation step, not as a replacement for local correspondence.  
   \- Adjust the point positions if needed

5\. Handle damage/artifact explicitly.  
   \- If the slice region is torn, distorted, collapsed, bubbled, folded, weakly stained, or otherwise damaged, say so explicitly before placing the point.  
   \- Do not treat the sharpest-looking local edge as trustworthy anatomy if the region appears damaged.  
   \- In damaged regions, prefer a stable intact neighboring wall segment or boundary feature over a distorted tip or artificial edge.

6\. Lock the point before moving on.  
   \- Before saving the point, explicitly check:  
     \- same local feature type,  
     \- same relative depth,  
     \- similar curvature,  
     \- similar neighboring contours,  
     \- similar boundary context,  
     \- and whether the point still makes sense after zooming out.  
   \- Then mark the point as confirmed and move to the next one.  
   \- Do not keep moving previously placed points unless you explicitly conclude that a prior feature definition was wrong.

Additional rules:  
\- Do not place points on the black background; only place them on the atlas/slice proper  
\- Do not assume left-right symmetry. The slice hemispheres may differ substantially.  
\- Do not rely on broad anatomical guesses without local confirmation.  
\- Use already confirmed points only as loose anchors for relative context, not as rigid constraints.

Task:  
\- Place \[n\] paired points on the outermost edge/border of the slice and atlas.  
\- Then place \[n\] paired points in the interior.  
\- For each point, report:  
  \- point number,  
  \- feature description,  
  \- why that feature matches locally,  
  \- and whether the zoomed-out anatomical context still supports the placement.  
