# ABBA

The Fiji connector lets an agent register sections inside ABBA, applying
changes as native, undoable ABBA registrations. It uses a separate Python
environment. Supported sessions use ABBA 0.24, Java 21, and the Allen Mouse
V3 or V3p1 atlas in the coronal plane; tilted cutting angles are supported.

## Install

1. Install [Miniforge](https://github.com/conda-forge/miniforge#download).
   Download and extract the LangSlice source, then run this in the folder
   containing `environment.yml`:

   ```bash
   conda env create -f environment.yml
   ```

   On Windows, use Miniforge Prompt or PowerShell with conda initialized.
   Keep the resulting `langslice` environment installed. If that name is
   already in use, add `-n langslice-fiji` and select that environment in Setup.

2. Build the connector using its
   [build instructions](https://github.com/greenpolo/LangSlice/blob/main/connectors/fiji/README.md#build).
   Copy `langslice-fiji-0.2.0.jar` into Fiji's `jars` folder and restart Fiji.
   Keep only one connector version installed. The connector is built against
   ABBA 0.24.1 and uses ABBA's dependencies.

3. Open **Plugins > LangSlice > LangSlice setup…** in Fiji, or **Setup…**
   in the LangSlice Registration dialog. Select the environment folder
   (not its Python executable or the source folder) and click
   **Check installation**.

4. Choose **Sign in with ChatGPT**, or **Save API key…** for an OpenAI or
   Gemini image model. Save setup. Credentials stay in LangSlice's user
   settings, outside the ABBA project; the setup check reports credential
   presence, not validated model access.

The connector and Python environment are installed from source. Setup is
available before Python is installed, so either component can be installed
first. The worker itself needs no JVM in the Python environment.

## Configure a registration

Import the images in ABBA, select the sections to process (none selected
means all), and open **Register > LangSlice Registration…**.

| Tab | Choices |
|---|---|
| Tasks | Positioning for order and atlas positions; Linear for in-plane alignment and optional cutting-angle estimation; Nonlinear for deformation. Each accepts notes to the agent. |
| Slices | Notes on damaged tissue, permission to mark damaged regions, and permission to overwrite existing transforms. |
| Preprocessing | Automatic or custom channel blending and contrast, snapshot pixel size, and a before/after preview. |

Section thickness and interval are prefilled from ABBA when available, or
from saved settings. Zero means unspecified: the agent infers what the
anatomy supports. There are no fixed lab protocol defaults. Nonlinear needs
a linear placement; when Linear is off and some sections have none, the
dialog offers to align them first or exclude them from Nonlinear.

Existing registrations keep their in-plane alignment unless **Allow the
agent to overwrite existing transforms** is enabled. Positions can still
move. A user's existing spline or BigWarp registration is protected.
Damage notes inform the agent; only atlas regions it marks are excluded
from fits.

**Let the agent drive preprocessing** exports every raw channel so its
channel tools can inspect and blend them. Those tools exist either way.
Use the preview to check contrast between anatomical regions.

## Run and review

With **ChatGPT**, choose the agent model and reasoning, then **Run**.
With **Claude**, configure the [MCP server](agents.md#mcp), choose
**Copy prompt**, and paste it into the MCP conversation. Keep ABBA's
progress window open to receive live updates.

For Nonlinear, choose an image model or **None (fit to the stain only)**.
The agent decides whether to use image-model tracing when it is available.
Model calls send section images to the selected provider. The estimated
cost appears at the bottom of the dialog.

Changes appear while the agent works. Each checkpoint is an ABBA undo step;
a section has at most one LangSlice affine and one BigWarp warp above it.
The warp can be opened in BigWarp. Avoid editing those sections'
registrations during a run: conflicting updates are refused and can be
retried with **Retry failed updates**.

**Stop run** stops the built-in agent and keeps completed changes. In Claude
mode, closing the window disconnects live updates; the MCP job continues
independently. Save with ABBA's normal Save command. Saved registrations
reopen without LangSlice installed.

**Show agent log** displays activity; **Save traces to** records a run for
inspection. **Open agent viewer** is available when ABBA was launched with
`langslice abba`. That optional command starts ABBA from Python and needs
the connector jar (`--connector-jar` or `LANGSLICE_CONNECTOR_JAR`) and Java
21, fetched on first start. An existing Fiji installation uses the connector
directly.

For troubleshooting, activate the environment and run `langslice version`.
Worker protocol and checkpoint handling are described in
[architecture](architecture.md#fiji-connector-and-worker).
