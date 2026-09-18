# LangSlice in your existing ABBA

The Fiji connector runs LangSlice in a separate Python environment. Open your
usual ABBA installation; its **Register > LangSlice** menu provides setup and
agent controls. You may install the Fiji plugin or the Python environment first.
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

Open **Register > LangSlice > LangSlice setup…**.

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
to use all), and open **Register > LangSlice > LangSlice agent…**. Review the
workflow and model before starting. **Linear agent** handles section order,
position and in-plane alignment; review its section spacing and enabled tasks.
**Nonlinear boundary refinement** refines sections at their existing atlas
positions. A progress window
reports activity and provides cancellation. Completed changes remain in ABBA
and can be saved using ABBA's normal state-save command.
This preview shows a text activity transcript; the older Python launcher's
comparison viewer is not included in the Fiji connector.

The connector currently requires the Allen Mouse V3p1 atlas in a flat coronal
session. Ordering/orientation is refused for slices with existing registrations;
turn that task off to refine an already registered stack. Avoid manually changing
the selected stack while an agent run is active. Cancelling stops the worker;
it does not automatically undo completed ABBA actions.

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
