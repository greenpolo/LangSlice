# LangSlice in your existing ABBA

The Fiji connector runs LangSlice in a separate Python environment. Open your
usual ABBA installation; its **Register > LangSlice > LangSlice Registration…**
entry opens LangSlice, and Fiji's **Plugins > LangSlice > LangSlice setup…** opens
setup. You may install the Fiji plugin or the Python environment first.
Account setup is done inside ABBA. Command-line login is optional.

**Distribution status:** this is a source-build preview. A public Fiji update
site and conda package have not been published. Do not use a purported
`conda install langslice` command yet. The environment-file installation below
installs the worker using pip inside conda. See the
[distribution notes](https://github.com/greenpolo/LangSlice/blob/main/packaging/README.md)
for the remaining conda packaging prerequisite.

## Install the Python environment

1. Install [Miniforge](https://github.com/conda-forge/miniforge#download).
2. Download and extract LangSlice's source archive. Open a terminal in the
   extracted folder containing `environment.yml`.
3. Run:

   ```bash
   conda env create -f environment.yml
   ```

This creates an environment named `langslice`, including its Python dependencies.
Leave it installed in that location. You can close the terminal when installation
finishes. You do not need Java, Maven or `abba-python` in this environment.

If an environment named `langslice` already exists, choose another name with
`conda env create -n langslice-fiji -f environment.yml` and select that environment
in Setup. This avoids changing an environment used by another project.

## Install the Fiji connector

Use an existing ABBA installation with Java 11 or newer. The connector targets
ABBA 0.11.0; compatibility with other ABBA versions needs testing.

For this source preview, build the connector following
[the connector instructions](https://github.com/greenpolo/LangSlice/blob/main/fiji-plugin/README.md),
copy `langslice-fiji-0.1.0.jar` into Fiji's `jars` folder, and restart Fiji before
opening ABBA. Keep only one version of the connector installed.

The planned public distribution is a LangSlice Fiji update site. Once published,
users will enable that site in Fiji's updater, apply changes, and restart Fiji;
no Java build will be needed. Standalone ABBA installations need an appropriate
plugin installation path; inclusion in ABBA's own installer is not required for
the Fiji route.

## Set up LangSlice in ABBA

Open Fiji's **Plugins > LangSlice > LangSlice setup…**, or click **Setup…** in the
LangSlice Registration dialog.

1. Select the environment folder containing LangSlice. The dialog discovers
   common conda environments; **Browse…** lets you select another location.
   Select the environment folder, not `python.exe`, an image folder or the
   LangSlice source folder.
2. Click **Check installation**. The plugin checks that the worker starts and
   speaks a compatible protocol.
3. Choose **Sign in with ChatGPT** and finish in your browser, or use
   **Save API key…** for OpenAI or Gemini. If the browser did not open, use
   **Open sign-in page**. API usage is billed by the provider. Model requests
   send section images to the selected provider.
4. Save setup. The environment location is remembered for future sessions.

Credentials stay in LangSlice's user settings, not the ABBA project. Setup
reports credential presence; it does not claim that an API key has been validated
online or that the account has access to every model. Environment variables
explicitly supplied to Fiji take precedence over saved API keys.

If the plugin is installed first, Setup still opens. Install the worker, select
its environment folder, and check again. If the environment is moved or removed,
choose its new location. Opening the agent before setup is complete takes you
through setup first.

## Run and save

Import sections into ABBA, select the sections to process (or leave none selected
to use all), and open **Register > LangSlice > LangSlice Registration…**. The
dialog has three tabs:

- **Tasks:** turn on **Positioning** (order, hemisphere flips and atlas positions;
  check the section thickness and interval, which are filled in from ABBA) and/or
  **Linear** (in-plane alignment). Each has a box for notes to the agent.
  **Nonlinear** is not yet available in ABBA.
- **Slices:** the slices that will be sent, with their number of ABBA
  registrations. Tick **Damaged** for torn or folded slices. By default, slices that
  already have ABBA registrations keep their in-plane alignment; their positions
  can still move. Tick **Allow the agent to overwrite existing transforms** to let
  the agent realign them too.
- **Preprocessing:** **Auto** usually works. **Custom** lets you weight each image
  channel and set local contrast (CLAHE). The preview shows each slice as exported
  and as the agent will see it. Try to maximize contrast between different regions.

Choose the model at the top, check the estimated cost at the bottom, and click
**Run**. Your choices are remembered for the next run.

Your ABBA session is not changed while the agent works. When the run finishes, the
result is applied to ABBA as one step, which ABBA's **Undo** reverts. If you click
**Stop run**, nothing is applied; the window then offers **Apply partial result to
ABBA** when the agent had saved at least one step. With **Show agent log** on, the
run window shows what the agent is doing; with it off, a small window shows only
the status and the final message. Save the project using ABBA's normal Save
command.

The connector currently requires the Allen Mouse V3p1 atlas in a flat coronal
session. Avoid changing the listed slices' registrations while a run is active: the
result is refused for a slice whose registrations changed outside the run.

## Optional terminal commands

For troubleshooting, these commands use the same installation and login:

```bash
conda activate langslice
langslice version
langslice login
```

The older `langslice abba` command starts ABBA from Python and is a separate
development route. It requires `langslice[abba]`, OpenJDK and Maven; it is not
needed by the Fiji connector.
