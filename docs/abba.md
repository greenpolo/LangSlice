# ABBA

:::{admonition} The Fiji connector is temporary
:class: warning

The Fiji connector is a stopgap. It will likely change, or be replaced,
in future versions of LangSlice and ABBA.
:::

The Fiji connector starts a LangSlice job from the sections you select in ABBA.
The registration runs in your AI app, and every change the agent saves
appears in ABBA as an ABBA registration you can undo, warps included.

It works with ABBA 0.25 (0.24.1 too) on Java 21, the Allen Mouse V3 or V3p1
atlas, and coronal sections.

## Install

1. [Install LangSlice](getting-started/install.md) and keep its environment:
   ABBA uses it.
2. Download `langslice-fiji-0.3.0.jar` from the
   [latest release](https://github.com/greenpolo/LangSlice/releases/latest).
   To build it yourself instead, with JDK 21 and
   [Maven](https://maven.apache.org): `git clone
   https://github.com/greenpolo/LangSlice.git`, then `mvn package` in
   `LangSlice/connectors/fiji`; the jar is in `target/`.
3. Copy the jar into Fiji's `jars` folder, remove any older
   `langslice-fiji` jar, and restart Fiji. Update LangSlice and the
   connector together: a connector refuses a LangSlice it cannot talk to.
4. The first time you use it, the connector asks for the LangSlice
   environment's folder. Choose it, click **Check installation**, then
   **Save**.

## Use

1. In ABBA, select the sections to register. With none selected, every
   section is used.
2. Choose **Register ‣ LangSlice…**. The LangSlice window opens with a message
   for your AI app, filled in from ABBA. Sections you already registered in
   ABBA keep their alignment and warps unless you untick **Keep … existing
   ABBA registrations**.
3. Add what you know, click **Copy**, and paste the message into a new
   conversation in your AI app.
4. Watch ABBA: each change the agent saves arrives as one ABBA undo step.
   Save with ABBA's own **ABBA - Save State**; the project reopens without
   LangSlice.

Choosing **Register ‣ LangSlice…** again continues the same job. Don't edit
sections in ABBA while the agent works on them: ABBA then refuses the agent's
later changes to those sections, and Fiji's Log window names them.

## Limits

- One cutting angle for the whole stack, as in ABBA.
- The agent works on snapshots of at most 3072 pixels across, not on your
  full-resolution images.
- If the LangSlice window does not open, choose **Plugins ‣ LangSlice ‣
  Choose Python environment…** and click **Check installation**.
