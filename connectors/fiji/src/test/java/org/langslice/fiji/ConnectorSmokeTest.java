package org.langslice.fiji;

import com.google.gson.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Offline executable integration checks, using a real fake-worker subprocess. */
public final class ConnectorSmokeTest {
    static void require(boolean condition, String message) { if (!condition) throw new AssertionError(message); }

    /** A worker that answers each method the connector calls, without a model or ABBA. */
    static final String FAKE_WORKER = String.join("\n",
        "#!/usr/bin/env python3",
        "import json,sys,time,zlib,struct",
        "r=json.loads(sys.stdin.readline()); m=r['method']; p=r.get('params') or {}",
        "def out(x): print(json.dumps(x)); sys.stdout.flush()",
        "def data(payload): out({'id':r['id'],'type':'event','event':{'kind':'data','payload':payload}})",
        "def result(x): out({'id':r['id'],'type':'result','result':x})",
        "def chunk(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)",
        "if m=='timeout':",
        " time.sleep(20)",
        "elif m=='setup.fail':",
        " out({'id':r['id'],'type':'error','error':'secret-key-must-not-escape'})",
        "elif m=='linear.estimate':",
        " if p.get('n_slices')==99: result({'low':2,'high':3.5,'unit':'percent_of_usage_window','basis':'test'})",
        " else: out({'id':r['id'],'type':'error','error':{'code':'runtime_error','message':'Runtime request handling failed','details':{'error':'No runs have been measured at this image resolution'}}})",
        "elif m=='preprocess.preview':",
        " rows=b''.join(b'\\x00'+bytes([0,120,240]) for y in range(2))",
        " png=b'\\x89PNG\\r\\n\\x1a\\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',3,2,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b'')",
        " open(p['output_path'],'wb').write(png); open(p['output_path']+'.json','w').write(json.dumps(p))",
        " result({'output_path':p['output_path'],'width':3,'height':2})",
        "elif m=='linear.run':",
        " data({'kind':'checkpoint','initial':True,'host_updates':[],'updates_since_start':[]})",
        " data({'kind':'log','message':'agent started'})",
        " data({'kind':'agent_event','event':{'kind':'tool_start','name':'look','target_ids':['a']}})",
        " data({'kind':'checkpoint','initial':False,'host_updates':[{'id':'a','position_mm':1}],'host_angles':{'pitch_deg':2.0,'yaw_deg':-1.0},'updates_since_start':[{'id':'a','position_mm':1}]})",
        " if p.get('stall'):",
        "  data({'kind':'checkpoint','initial':False,'host_updates':[{'id':'a','position_mm':2}],'updates_since_start':[{'id':'a','position_mm':2}]})",
        "  time.sleep(20)",
        " result({'state':{'submitted':True},'output_dir':'out','final_updates':[{'id':'a','position_mm':3}],'echo':p})",
        "else:",
        " out({'id':None,'type':'event'})",
        " data({'kind':'checkpoint'})",
        " result({'protocol_version':1})",
        "");

    static final class Recorder implements AgentRunner.Progress {
        final List<String> lines = Collections.synchronizedList(new ArrayList<>());
        public void line(String text) { lines.add(text); }
        public void fragment(String text) { lines.add(text); }
        public void status(String text) { lines.add("status: " + text); }
    }

    static void hostChannel() throws Exception {
        Recorder progress = new Recorder();
        AtomicReference<JsonObject> applied = new AtomicReference<>();
        try (McpHostChannel channel = new McpHostChannel()) {
            JsonObject settings = channel.settings();
            java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newSingleThreadExecutor();
            try {
                java.util.concurrent.Future<JsonObject> received = executor.submit(() -> channel.receive(
                        event -> AgentRunner.handleEvent(event, progress, new AtomicBoolean(), new int[]{0}, applied::set)));
                int port = settings.get("port").getAsInt();
                try (java.net.Socket wrong = new java.net.Socket("127.0.0.1", port)) {
                    wrong.getOutputStream().write("{\"token\":\"wrong\"}\n".getBytes(StandardCharsets.UTF_8));
                    wrong.setSoTimeout(3000);
                    require(wrong.getInputStream().read() == -1, "Wrong token is rejected");
                }
                try (java.net.Socket socket = new java.net.Socket("127.0.0.1", port)) {
                    JsonObject auth = new JsonObject(); auth.add("token", settings.get("token"));
                    String messages = auth + "\n"
                            + "{\"type\":\"event\",\"event\":{\"kind\":\"data\",\"payload\":{\"kind\":\"checkpoint\",\"initial\":false,\"host_updates\":[{\"id\":\"a\",\"position_mm\":2}],\"updates_since_start\":[{\"id\":\"a\",\"position_mm\":2}]}}}\n"
                            + "{\"type\":\"result\",\"result\":{\"final_updates\":[]}}\n";
                    socket.getOutputStream().write(messages.getBytes(StandardCharsets.UTF_8));
                    require(received.get(5, java.util.concurrent.TimeUnit.SECONDS).has("final_updates"), "Authenticated final result");
                    require(applied.get().getAsJsonArray("host_updates").get(0).getAsJsonObject().get("position_mm").getAsInt() == 2,
                            "Live host update reaches common apply callback");
                }
            } finally { executor.shutdownNow(); }
        }
    }

    public static void main(String[] args) throws Exception {
        hostChannel();
        Path prefix = Files.createTempDirectory("langslice connector space ");
        boolean windows = System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win");
        Path python = windows ? prefix.resolve("python.exe") : prefix.resolve("bin/python");
        Files.createDirectories(python.getParent());
        Files.write(python, FAKE_WORKER.getBytes(StandardCharsets.UTF_8)); python.toFile().setExecutable(true);
        require(EnvironmentDiscovery.python(prefix).equals(python), "Prefix resolves python");
        require(EnvironmentDiscovery.command(prefix).get(0).equals(python.toString()), "Prefix with spaces stays a single argv element");
        EnvironmentDiscovery.save(prefix);
        require(prefix.equals(EnvironmentDiscovery.saved()), "Remember environment path with spaces");
        require(EnvironmentDiscovery.discover().contains(prefix), "Saved environment is rediscovered");
        Path launcher = Files.createTempDirectory("langslice launcher ");
        Path launcherPython = windows ? launcher.resolve("python.exe") : launcher.resolve("bin/python");
        Files.createDirectories(launcherPython.getParent()); Files.write(launcherPython, new byte[0]);
        System.setProperty(EnvironmentDiscovery.PROPERTY, launcher.toString());
        try {
            require(EnvironmentDiscovery.current().equals(launcher.toAbsolutePath().normalize()) && prefix.equals(EnvironmentDiscovery.saved()),
                    "langslice abba's environment runs this session's workers; the saved choice is kept");
            require(EnvironmentDiscovery.discover().get(0).equals(launcher.toAbsolutePath().normalize()), "Setup lists the launcher's environment first");
            System.setProperty(EnvironmentDiscovery.PROPERTY, launcher.resolve("missing").toString());
            require(prefix.equals(EnvironmentDiscovery.current()), "A launcher environment without Python falls back to the saved one");
        } finally {
            System.clearProperty(EnvironmentDiscovery.PROPERTY);
            Files.walk(launcher).sorted(Comparator.reverseOrder()).forEach(path -> path.toFile().delete());
        }
        if (windows) {
            // A POSIX shebang makes the fake worker executable on Unix only.
            // Exercise discovery, settings, events and the SciJava menu here;
            // Python's installed worker protocol is covered by the Windows pytest job.
            eventChecks();
            settingsChecks();
            menuChecks();
            Files.walk(prefix).sorted(Comparator.reverseOrder()).forEach(path -> path.toFile().delete());
            System.out.println("Connector Windows discovery and command smoke checks passed.");
            return;
        }
        AtomicInteger events = new AtomicInteger();
        try (WorkerClient client = new WorkerClient(prefix)) {
            JsonObject result = client.request("hello", new JsonObject(), e -> events.incrementAndGet(), Duration.ofSeconds(5));
            require(result.get("protocol_version").getAsInt() == 1 && events.get() == 1, "Events and results traverse subprocess pipe");
        }
        try (WorkerClient client = new WorkerClient(prefix)) {
            try { client.request("setup.fail", new JsonObject(), null, Duration.ofSeconds(5)); throw new AssertionError("Expected failure"); }
            catch (IOException expected) { require(!expected.getMessage().contains("secret-key"), "Setup errors redact provider secrets"); }
        }
        long start = System.nanoTime();
        try (WorkerClient client = new WorkerClient(prefix)) {
            try { client.request("timeout", new JsonObject(), null, Duration.ofMillis(200)); throw new AssertionError("Expected timeout"); }
            catch (IOException expected) { require(expected.getMessage().contains("timed out"), "Timeout is actionable"); }
        }
        require((System.nanoTime() - start) < 5_000_000_000L, "Silent worker cannot block indefinitely");

        eventChecks();
        runChecks(prefix);
        settingsChecks();
        snapshotChecks(prefix);
        estimateChecks(prefix);
        menuChecks();
        Files.walk(prefix).sorted(Comparator.reverseOrder()).forEach(path -> path.toFile().delete());
        System.out.println("Connector smoke checks passed.");
    }

    /** Every saved checkpoint goes to the live hook as it arrives (ChatGPT mode too); listeners get events and checkpoints. */
    static void runChecks(Path prefix) throws Exception {
        JsonObject request = new JsonObject(); request.addProperty("image_folder", "x");
        request.add("locked", JsonParser.parseString("[\"section_0002.tif\"]"));
        request.add("damaged", JsonParser.parseString("{\"section_0001.tif\":\"torn\"}"));
        request.add("existing_warp", JsonParser.parseString("[\"section_0002.tif\"]"));
        List<JsonObject> live = Collections.synchronizedList(new ArrayList<>());
        List<String> heard = Collections.synchronizedList(new ArrayList<>());
        java.util.function.Consumer<String> listener = heard::add;
        LangSliceEvents.addListener(listener);
        Recorder progress = new Recorder();
        JsonObject result;
        try (WorkerClient client = new WorkerClient(prefix)) {
            result = AgentRunner.runLinear(client, request, progress, new AtomicBoolean(), live::add);
        } finally { LangSliceEvents.flush(); LangSliceEvents.removeListener(listener); }
        require(live.size() == 1, "The initial checkpoint is never applied; every later one is, live: " + live.size());
        JsonObject applied = live.get(0);
        require(applied.getAsJsonArray("host_updates").get(0).getAsJsonObject().get("position_mm").getAsInt() == 1
                && applied.getAsJsonObject("host_angles").get("pitch_deg").getAsDouble() == 2.0, "Live payload carries host rows and host_angles");
        require(result.getAsJsonObject("echo").get("locked").equals(request.get("locked"))
                && result.getAsJsonObject("echo").get("damaged").equals(request.get("damaged"))
                && result.getAsJsonObject("echo").get("existing_warp").equals(request.get("existing_warp")), "Locked, damaged and existing_warp reach the worker");
        require(progress.lines.contains("agent started") && progress.lines.stream().anyMatch(l -> l.contains("Saved steps: 1") && l.contains("live in ABBA")),
                "Run log and live status reach the window");
        require(heard.stream().filter(m -> m.contains("\"kind\":\"checkpoint\"")).count() == 2, "Both checkpoints reach listeners");
        require(heard.stream().anyMatch(m -> m.contains("\"kind\":\"agent_event\"") && m.contains("\"look\"")), "Agent events reach listeners");

        // A stop ends the run; checkpoints before it were already applied live.
        JsonObject stall = new JsonObject(); stall.addProperty("stall", true);
        List<JsonObject> stopped = Collections.synchronizedList(new ArrayList<>());
        AtomicBoolean stopping = new AtomicBoolean();
        try (WorkerClient client = new WorkerClient(prefix)) {
            Thread stopper = new Thread(() -> {
                long deadline = System.nanoTime() + 10_000_000_000L;
                while (System.nanoTime() < deadline && stopped.size() < 2) {
                    try { Thread.sleep(20); } catch (InterruptedException e) { return; }
                }
                stopping.set(true); client.close();
            });
            stopper.start();
            try { AgentRunner.runLinear(client, stall, new Recorder(), stopping, stopped::add); throw new AssertionError("Expected the stop to end the run"); }
            catch (Exception expected) { }
            stopper.join();
        }
        require(stopped.size() == 2 && stopped.get(1).getAsJsonArray("host_updates").get(0).getAsJsonObject().get("position_mm").getAsInt() == 2,
                "Every checkpoint before the stop was applied live");
    }

    /** Listeners: delivery in order, off the caller's thread; a failing listener affects nothing else. */
    static void eventChecks() throws Exception {
        require(!LangSliceEvents.hasListeners(), "No listener before ABBA is started from Python");
        List<String> got = Collections.synchronizedList(new ArrayList<>());
        java.util.function.Consumer<String> broken = text -> { throw new IllegalStateException("listener bug"); };
        java.util.function.Consumer<String> good = got::add;
        LangSliceEvents.addListener(broken); LangSliceEvents.addListener(good);
        try {
            require(LangSliceEvents.hasListeners(), "A registered listener makes the viewer available");
            JsonObject body = new JsonObject(); body.addProperty("n", 1);
            LangSliceEvents.publish("agent_event", body);
            body.addProperty("n", 2);
            LangSliceEvents.publish("agent_event", body);
            LangSliceEvents.flush();
            require(got.size() == 2 && got.get(0).contains("\"n\":1") && got.get(1).contains("\"n\":2") && got.get(0).contains("\"kind\":\"agent_event\""),
                    "Messages arrive in order despite a failing listener: " + got);
            RegistrationDialog.Host host = new RegistrationDialog.Host() {
                public java.awt.image.BufferedImage[] preview(int s, List<Integer> c, JsonObject p, double px) { return null; }
                public JsonObject estimate(JsonObject p) { return null; }
                public JsonObject status() { return null; }
                public void setup() { }
                public void run(RegistrationSettings s, Map<Integer, String> d, Set<Integer> n) { }
            };
            require(host.viewerAvailable(), "Viewer offered while a listener is registered");
        } finally { LangSliceEvents.removeListener(broken); LangSliceEvents.removeListener(good); }
        require(!LangSliceEvents.hasListeners(), "Listeners can be removed");
        JsonObject refused = JsonParser.parseString("{\"kind\":\"tool_end\",\"name\":\"elastix_affine\",\"response\":{\"status\":\"error\","
                + "\"error\":\"LOCKED\",\"message\":\"s1 is locked\"}}").getAsJsonObject();
        require(AgentRunner.agentText(refused).equals("\ns1 is locked\n"), "A failing tool's reason reaches the run log (both modes)");
        refused.addProperty("kind", "tool_result");
        require(AgentRunner.agentText(refused).isEmpty(), "The ADK's own tool_result does not repeat it");
    }

    /** Dialog choices become the contract's spec, preprocessing and exported channel order, and persist. */
    static void settingsChecks() throws Exception {
        RegistrationSettings s = new RegistrationSettings();
        s.model = RegistrationSettings.modelId("gpt-6-luna"); s.reasoning = "high"; s.resolution = "medium";
        s.positioning = true; s.linear = true; s.affine = false; s.maxParallel = 2; s.agentDamage = false;
        s.positionNotes = " thick sections "; s.linearNotes = "tears ventral"; s.cue = "ink right";
        JsonObject spec = s.spec();
        require(spec.getAsJsonArray("tasks").toString().equals("[\"reorder\",\"position\",\"transform\"]"), "Positioning = reorder+position, Linear = transform");
        require(spec.get("model").getAsString().equals("openai-oauth/gpt-6-luna") && spec.get("reasoning").getAsString().equals("high"), "Model id and reasoning");
        require(spec.get("image_resolution").getAsString().equals("medium") && !spec.get("agent_damage").getAsBoolean(), "Resolution and agent damage flag");
        JsonObject transform = spec.getAsJsonObject("transform");
        require(!transform.get("automatic").getAsBoolean() && transform.get("interactive").getAsBoolean() && !transform.has("elastix")
                && !transform.get("angles").getAsBoolean() && transform.get("max_parallel").getAsInt() == 2 && transform.get("notes").getAsString().equals("tears ventral"), "Linear block");
        require(transform.get("flip").getAsBoolean() && transform.get("hemisphere_cue").getAsString().equals("ink right") && !spec.has("reorder"),
                "Flip and hemisphere cue belong to Linear (transform.*), never reorder.*");
        require(spec.getAsJsonObject("position").get("notes").getAsString().equals("thick sections"), "Positioning block");
        require(!spec.has("inputs"), "Damaged slices travel as a request parameter, never in the spec");
        require(!spec.get("agent_preprocessing").getAsBoolean() && !spec.getAsJsonArray("tasks").toString().contains("nonlinear"), "Off by default");
        s.angles = true;
        require(s.spec().getAsJsonObject("transform").get("angles").getAsBoolean(), "Slice angle estimation sends transform.angles");
        s.positioning = false;
        require(spec(s).equals("[\"transform\"]"), "Linear alone");
        s.linear = false;
        require(s.problem(1) != null, "At least one task is required");
        s.nonlinear = true; s.nonlinearNotes = " hippocampus torn "; s.engine = "ants"; s.imageProvider = "none"; s.imageModel = null;
        require(s.problem(1) == null && spec(s).equals("[\"nonlinear\"]"), "Nonlinear alone is a valid run");
        JsonObject deform = s.spec().getAsJsonObject("nonlinear");
        require(deform.get("provider").getAsString().equals("none") && deform.get("image_model").isJsonNull()
                && deform.get("engine").getAsString().equals("ants") && deform.get("notes").getAsString().equals("hippocampus torn"),
                "Nonlinear block: no image model, engine and notes are sent");
        s.imageProvider = "gemini-api"; s.imageModel = "gemini-3-pro-image";
        require(s.spec().getAsJsonObject("nonlinear").get("image_model").getAsString().equals("gemini-3-pro-image") && s.usesImageModel(), "Image provider and model");
        s.linear = true; s.nonlinear = false; s.imageProvider = "openai-oauth"; s.imageModel = "gpt-image-2"; s.engine = "either";
        require(s.preprocessing(3).toString().equals("{\"mode\":\"auto\"}") && s.exportChannels(3).equals(Arrays.asList(0, 1, 2)), "Auto exports every channel");
        s.custom = true; s.weights = new double[]{1, 0, 0.5}; s.clahe = false; s.strength = "high";
        require(s.exportChannels(3).equals(Arrays.asList(0, 2)), "Custom exports only weighted channels, in channel order");
        JsonObject preprocessing = s.preprocessing(3);
        require(preprocessing.get("mode").getAsString().equals("custom") && !preprocessing.get("clahe").getAsBoolean()
                && preprocessing.get("clahe_strength").getAsString().equals("high")
                && preprocessing.getAsJsonArray("channel_weights").toString().equals("[1.0,0.5]"), "Custom weights follow the page order");
        s.agentPreprocessing = true;
        require(s.exportChannels(3).equals(Arrays.asList(0, 1, 2)) && s.spec().get("agent_preprocessing").getAsBoolean()
                && s.preprocessing(3).getAsJsonArray("channel_weights").toString().equals("[1.0,0.0,0.5]"),
                "Agent-driven preprocessing exports every channel, weight 0 included");
        s.agentPreprocessing = false;
        require(AbbaHostSession.exportedNames(Arrays.asList(0, 2), Arrays.asList("DAPI", "", "DAPI")).toString().equals("[\"DAPI\",\"DAPI (2)\"]")
                && AbbaHostSession.exportedNames(Arrays.asList(0, 1, 2), Arrays.asList("x", "x", "")).toString().equals("[\"x\",\"x (2)\",\"ch3\"]"),
                "channel_names follow the exported pages, never empty or repeated");
        s.weights = new double[]{0, 0, 0};
        require(s.problem(3) != null, "All-zero weights are refused");
        require(new RegistrationSettings().weightsFor(2).length == 2, "Saved weights adapt to the channel count");
        s.weights = new double[]{0.2, 0.7, 1};
        s.nonlinear = true; s.angles = true; s.agentPreprocessing = true; s.viewer = true; s.engine = "elastix";
        s.imageProvider = "none"; s.imageModel = null; s.nonlinearNotes = "notes";
        s.saveTraces = true; s.traceDir = " ";
        require(s.problem(3) != null, "Saving traces needs a folder");
        s.traceDir = "/tmp/langslice traces";
        require(s.problem(3) == null, "A trace folder with a space is accepted");
        java.util.prefs.Preferences node = java.util.prefs.Preferences.userRoot().node("langslice-smoke-" + System.nanoTime());
        s.save(node);
        RegistrationSettings back = RegistrationSettings.load(node);
        require(!back.fresh && back.spec().equals(s.spec()) && back.preprocessing(3).equals(s.preprocessing(3)) && back.pixelSize == s.pixelSize
                && back.overwrite == s.overwrite && back.showLog == s.showLog && back.viewer
                && back.imageProvider.equals("none") && back.imageModel == null && back.engine.equals("elastix")
                && back.saveTraces && back.traceDir.equals(s.traceDir), "Every choice persists between runs");
        node.removeNode();
        JsonObject status = JsonParser.parseString("{\"providers\":{\"openai-oauth\":{\"configured\":true,\"agent_models\":[\"openai-oauth/gpt-6-sol\"],"
                + "\"default_agent_model\":\"openai-oauth/gpt-6-sol\",\"default_image_model\":\"gpt-image-2\"}}}").getAsJsonObject();
        require(RegistrationSettings.agentModels(status).equals(Collections.singletonList("openai-oauth/gpt-6-sol")), "Models come from setup.status");
        require(RegistrationSettings.accountDefault(status, "default_image_model").equals("gpt-image-2")
                && RegistrationSettings.accountDefault(new JsonObject(), "default_agent_model") == null, "Account defaults");
        require(RegistrationSettings.signedIn(status) && !RegistrationSettings.signedIn(new JsonObject()), "Account status");
        require(RegistrationSettings.modelLabel("openai-oauth/gpt-5.6-sol").equals("gpt-5.6-sol"), "Model labels drop the account prefix");
        require(new RegistrationSettings().problem(1) != null, "No agent model until the dialog fills in the worker's default");
        // Image models in the worker's order, with None (fit to the stain only) as the worker places it.
        JsonObject current = JsonParser.parseString("{\"image_models\":[{\"provider\":\"openai-oauth\",\"label\":\"ChatGPT image lane\",\"connected\":true,"
                + "\"models\":[\"gpt-image-2\"],\"default_model\":\"gpt-image-2\"},{\"provider\":\"gemini-api\",\"label\":\"Gemini API\",\"connected\":false,"
                + "\"models\":[\"g1\",\"g2\"],\"default_model\":\"g1\"},{\"provider\":\"openai-api\",\"label\":\"OpenAI API\",\"connected\":true,\"models\":[],"
                + "\"default_model\":\"img-1\"},{\"provider\":\"none\",\"label\":\"None\",\"connected\":true,\"models\":[],\"default_model\":null}]}").getAsJsonObject();
        List<RegistrationSettings.ImageChoice> given = RegistrationSettings.imageChoices(current);
        require(given.size() == 5 && given.get(0).label.equals("ChatGPT image lane: gpt-image-2") && given.get(2).matches("gemini-api", "g2")
                && given.get(2).toString().endsWith("(not set up)") && given.get(3).matches("openai-api", "img-1")
                && given.get(4).label.equals("None (fit to the stain only)"), "setup.status image_models list: " + given);
        require(RegistrationSettings.imageChoices(new JsonObject()).isEmpty(), "No list, no choices");
    }

    private static String spec(RegistrationSettings s) { return s.spec().getAsJsonArray("tasks").toString(); }

    /** Multi-page snapshots keep the requested channel order; the preview request reaches the worker. */
    static void snapshotChecks(Path prefix) throws Exception {
        ij.process.ByteProcessor first = new ij.process.ByteProcessor(4, 3), second = new ij.process.ByteProcessor(4, 3);
        first.set(10); second.set(200);
        ij.ImagePlus image = AbbaHostSession.stack("section", Arrays.asList(first, second), null);
        Path folder = Files.createTempDirectory("langslice snapshot ");
        Path tif = folder.resolve("section_0001.tif");
        AbbaHostSession.save(image, tif);
        ij.ImagePlus reopened = new ij.io.Opener().openImage(tif.toString());
        require(reopened.getStackSize() == 2 && reopened.getStack().getProcessor(1).get(0, 0) == 10
                && reopened.getStack().getProcessor(2).get(0, 0) == 200, "Multi-page TIFF keeps channel order");
        ij.process.ShortProcessor deep = new ij.process.ShortProcessor(4, 3); deep.set(4000);
        require(AbbaHostSession.stack("mixed", Arrays.asList(first, deep), null).getBitDepth() == 32, "Mixed pixel types share one float TIFF");
        java.awt.image.BufferedImage shown = AbbaHostSession.display(image);
        require(shown.getWidth() == 4 && shown.getHeight() == 3, "Before-image display keeps the snapshot size");
        JsonObject preprocessing = JsonParser.parseString("{\"mode\":\"custom\",\"clahe\":true,\"clahe_strength\":\"low\",\"channel_weights\":[1.0,0.5]}").getAsJsonObject();
        Path png = folder.resolve("after.png");
        java.awt.image.BufferedImage after;
        try (WorkerClient client = new WorkerClient(prefix)) { after = AgentRunner.requestPreview(client, tif, preprocessing, png); }
        require(after.getWidth() == 3 && after.getHeight() == 2, "Preview PNG is read back");
        require(!Files.exists(png), "Preview PNG is removed after reading");
        JsonObject sent = JsonParser.parseString(new String(Files.readAllBytes(folder.resolve("after.png.json")), StandardCharsets.UTF_8)).getAsJsonObject();
        require(sent.get("image_path").getAsString().equals(tif.toString()) && sent.get("preprocessing").equals(preprocessing), "Preview request carries snapshot and preprocessing");
        Files.walk(folder).sorted(Comparator.reverseOrder()).forEach(path -> path.toFile().delete());
    }

    /** The estimate line: a number, or the worker's plain reason when it gives none. */
    static void estimateChecks(Path prefix) throws Exception {
        JsonObject params = new JsonObject(); params.addProperty("n_slices", 99);
        try (WorkerClient client = new WorkerClient(prefix)) {
            String text = RegistrationDialog.costText(client.request("linear.estimate", params, null, Duration.ofSeconds(5)));
            require(text.equals("Estimated cost: 2–3.5% of your ChatGPT usage window"), "Estimate text: " + text);
        }
        params.addProperty("n_slices", 98);
        RegistrationSettings medium = new RegistrationSettings(); medium.resolution = "medium";
        try (WorkerClient client = new WorkerClient(prefix)) {
            client.request("linear.estimate", params, null, Duration.ofSeconds(5)); throw new AssertionError("Expected a refusal");
        } catch (IOException refused) {
            String text = RegistrationDialog.refusalText(refused, medium);
            require(text.equals("Estimated cost: no estimate (only runs at Low image resolution have been measured)"), "Refusal text: " + text);
        }
        require(RegistrationDialog.costText(JsonParser.parseString("{\"low\":null,\"high\":null,\"unit\":\"percent_of_usage_window\",\"available\":false,"
                + "\"basis\":\"no runs measured at Medium resolution\"}").getAsJsonObject()).equals("Estimated cost: no estimate (no runs measured at Medium resolution)"),
                "A worker without a number shows its plain reason");
        RegistrationSettings deform = new RegistrationSettings(); deform.nonlinear = true;
        require(RegistrationDialog.refusalText(new IOException("{\"code\":\"runtime_error\",\"message\":\"Runtime request handling failed\"}"), deform)
                .contains("Nonlinear"), "A refusal without a reason still says why");
    }

    /** ABBA's Register menu gets one LangSlice submenu with one callable entry. */
    @SuppressWarnings("unchecked")
    static void menuChecks() throws Exception {
        // Plugin annotations are compiled into SciJava's discovery index, not manually registered at runtime.
        java.io.InputStream index = LangSliceService.class.getResourceAsStream("/META-INF/json/org.scijava.plugin.Plugin");
        require(index != null, "SciJava plugin index exists");
        String plugins = new String(index.readAllBytes(), StandardCharsets.UTF_8); index.close();
        require(plugins.contains("RegistrationCommand") && !plugins.contains("AgentCommand"), "Plugin index lists only current commands");
        try (org.scijava.Context context = new org.scijava.Context(org.scijava.plugin.PluginService.class, org.scijava.command.CommandService.class, LangSliceService.class)) {
            org.scijava.command.CommandService commands = context.getService(org.scijava.command.CommandService.class);
            org.scijava.command.CommandInfo registration = commands.getCommand(LangSliceService.REGISTRATION);
            require(registration != null, "ABBA menu string resolves to a callable command");
            require(registration.createInstance() instanceof RegistrationCommand, "Command alias constructs the registration command");
            require(registration.getInput("mp") != null, "Alias preserves live-session parameter injection");
            java.lang.reflect.Field field = ch.epfl.biop.atlas.aligner.MultiSlicePositioner.class.getDeclaredField("externalRegistrationPluginsUI");
            field.setAccessible(true);
            Map<String, List<String>> ui = (Map<String, List<String>>) field.get(null);
            require(ui.get(LangSliceService.MENU).equals(Collections.singletonList(LangSliceService.REGISTRATION)), "One Register entry: " + ui);
            require(!ui.containsKey("LangSlice connector"), "Old submenu is gone");
            // ABBA builds the path as "Register>" + string.
            require(("Register>" + LangSliceService.REGISTRATION).equals("Register>LangSlice Registration…"), "A plain Register entry, not a submenu");
        }
    }
}
