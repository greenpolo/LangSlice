# LangSlice Fiji connector

This independently distributed Java plugin connects an existing ABBA installation
with a separately installed LangSlice Python environment. Python and model accounts
are not needed to load the plugin or open Setup.

The connector targets ABBA **0.11.0** and Java **11 or newer**. Newer ABBA versions
must be checked before advertising compatibility: upstream has changed source
processor package names. Runtime dependencies are supplied by ABBA/Fiji and are
not bundled into the plugin JAR.

## Build

With JDK 11+ and Maven:

```bash
cd fiji-plugin
mvn package dependency:build-classpath -Dmdep.outputFile=target/test-classpath.txt
```

After Maven has written the resolved dependency classpath, subsequent development
builds can run offline:

```bash
python fiji-plugin/build.py --java-home /path/to/jdk --test
```

The offline smoke checks start real fake-worker subprocesses to exercise event and
result delivery, timeout handling, secret-safe authentication errors, and paths
containing spaces. They also check the generated SciJava discovery index and native
affine/spline registration geometry and serialization. The offline build uses the
exact Maven-resolved classpath, never every JAR found in the Maven cache. They do
not replace live Fiji testing of registration, cancellation, undo and saved state.

Copy `target/langslice-fiji-0.1.0.jar` into your Fiji `jars` folder and restart
Fiji. This manual JAR installation is the development route; a public Fiji update
site has not been published. Do not copy the Maven dependency jars into Fiji.

## Setup and lifecycle

Open ABBA, then **Register → LangSlice → LangSlice setup…**. The same setup is
available from Fiji's **Plugins → LangSlice** menu. Choose the conda environment,
check the installation, and connect a model account. The browser completes ChatGPT
sign-in. API keys are sent through the worker's stdin, never command-line arguments,
and are saved by LangSlice. A saved credential is explicitly distinguished from an
account validated online.

Environment discovery checks conda's registry and common installation folders;
Browse accepts an environment folder. A remembered environment is rechecked when
launching the agent. When available, `conda run --prefix` supplies activation variables
without requiring an open terminal. Otherwise the environment's Python is invoked
directly. Worker processes and their descendants are terminated on cancellation.

`LangSliceService` registers named commands with ABBA's existing registration UI
registry before the ABBA window is built. The registry supports the desired nested
menu and passes the live session as `mp`. No `PyCommandBuilder`, `abba_python`, or
upstream ABBA modifications are required.

The connector uses protocol version 1. A request is one JSON line sent to
`python -m langslice serve --stdio`; the worker streams events and then a result or
error. Images and registration outputs remain local files. No local HTTP server or
listening port is required for the connector itself.
