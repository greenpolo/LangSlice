package org.langslice.fiji;

import com.google.gson.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/** Offline executable checks, using a real fake-worker subprocess that speaks protocol 3. */
public final class ConnectorSmokeTest {
    static void require(boolean condition, String message) { if (!condition) throw new AssertionError(message); }

    /**
     * A worker that answers each method the connector calls, without a model or ABBA; with "gui" among its
     * arguments it stands in for the LangSlice window.
     */
    static final String FAKE_WORKER = String.join("\n",
        "#!/usr/bin/env python3",
        "import json,sys,time",
        "if 'gui' in sys.argv:",
        " print('argv ' + json.dumps(sys.argv[sys.argv.index('gui'):])); sys.stdout.flush()",
        " if any(a.endswith('fails') for a in sys.argv): print('gui: error: unrecognized arguments: --tab'); sys.exit(2)",
        " print('LangSlice window: http://127.0.0.1:1/'); sys.stdout.flush(); time.sleep(30); sys.exit(0)",
        "r=json.loads(sys.stdin.readline()); m=r['method']; p=r.get('params') or {}",
        "def out(x): print(json.dumps(x)); sys.stdout.flush()",
        "def data(payload): out({'id':r['id'],'type':'event','event':{'kind':'data','payload':payload}})",
        "def result(x): out({'id':r['id'],'type':'result','result':x})",
        "if m=='timeout':",
        " time.sleep(20)",
        "elif m=='setup.status':",
        " sys.stderr.write('secret-key-must-not-escape\\n'); sys.stderr.flush()",
        " result({'version':'9.9','protocol_version':3,'python_executable':sys.executable,'environment_prefix':'x'})",
        "elif m=='abba.prepare':",
        " sys.stderr.write('secret-key-must-not-escape\\n'); sys.stderr.flush()",
        " out({'id':r['id'],'type':'error','error':{'code':'runtime_error','message':'Runtime request handling failed','details':{'error':'this job already ran'}}})",
        "elif m=='abba.follow':",
        " folder=p['job_folder']",
        " sys.stderr.write('follower warning\\n'); sys.stderr.flush()",
        " if folder.endswith('waiting'): data({'kind':'waiting'})",
        " else:",
        "  data({'kind':'checkpoint','initial':False,'host_updates':[{'id':'a','position_mm':1}],'host_angles':{'pitch_deg':2.0,'yaw_deg':-1.0}})",
        "  data({'kind':'log','message':'section a: warp sent'})",
        "  data({'kind':'checkpoint','initial':False,'host_updates':[{'id':'b','position_mm':2},{'id':'bad','position_mm':3}]})",
        " if folder.endswith('stubborn'): time.sleep(60)",
        " sys.stdin.read()",
        " open(folder+'/ended','w').write('stdin closed')",
        " result({})",
        "else:",
        " out({'id':None,'type':'event'})",
        " data({'kind':'checkpoint'})",
        " result({'protocol_version':3})",
        "");

    public static void main(String[] args) throws Exception {
        Path prefix = Files.createTempDirectory("langslice connector space ");
        boolean windows = System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win");
        Path python = windows ? prefix.resolve("python.exe") : prefix.resolve("bin/python");
        Files.createDirectories(python.getParent());
        Files.write(python, FAKE_WORKER.getBytes(StandardCharsets.UTF_8)); python.toFile().setExecutable(true);
        require(EnvironmentDiscovery.python(prefix).equals(python), "Prefix resolves python");
        List<String> serve = EnvironmentDiscovery.command(prefix, "serve", "--stdio");
        require(serve.get(0).equals(python.toString()) && serve.subList(1, serve.size()).equals(Arrays.asList("-u", "-m", "langslice", "serve", "--stdio")),
                "Prefix with spaces stays a single argv element; the worker is langslice serve --stdio: " + serve);
        require(EnvironmentDiscovery.command(prefix, "gui", "/a b/langslice", "--tab", "prompt").subList(4, 8).equals(Arrays.asList("gui", "/a b/langslice", "--tab", "prompt")),
                "The window's command carries its arguments");
        EnvironmentDiscovery.save(prefix);
        require(prefix.equals(EnvironmentDiscovery.saved()), "Remember environment path with spaces");
        require(EnvironmentDiscovery.discover().get(0).equals(prefix.toAbsolutePath().normalize()), "The saved environment is listed first");
        projectKeyChecks();
        menuChecks();
        snapshotChecks();
        if (windows) {
            // A POSIX shebang makes the fake worker executable on Unix only.
            delete(prefix);
            System.out.println("Connector Windows discovery, project and menu smoke checks passed.");
            return;
        }
        transportChecks(prefix);
        followerChecks(prefix);
        windowChecks(prefix);
        delete(prefix);
        System.out.println("Connector smoke checks passed.");
    }

    static void delete(Path folder) throws IOException {
        try (java.util.stream.Stream<Path> files = Files.walk(folder)) { files.sorted(Comparator.reverseOrder()).forEach(path -> path.toFile().delete()); }
    }

    /** Events and results, timeouts, and stderr: drained for every request but the follower's, never in an error. */
    static void transportChecks(Path prefix) throws Exception {
        AtomicInteger events = new AtomicInteger();
        List<String> stderr = Collections.synchronizedList(new ArrayList<>());
        try (WorkerClient client = new WorkerClient(prefix)) {
            client.stderrLog = stderr::add;
            JsonObject result = client.request("hello", new JsonObject(), e -> events.incrementAndGet(), Duration.ofSeconds(5));
            require(result.get("protocol_version").getAsInt() == 3 && events.get() == 1, "Events and results traverse the subprocess pipe");
        }
        try (WorkerClient client = new WorkerClient(prefix)) {
            client.stderrLog = stderr::add;
            JsonObject status = client.request("setup.status", new JsonObject(), null, Duration.ofSeconds(5));
            require(status.get("protocol_version").getAsInt() == OpenLangSliceCommand.PROTOCOL_VERSION && status.has("python_executable"), "setup.status v3");
        }
        try (WorkerClient client = new WorkerClient(prefix)) {
            client.stderrLog = stderr::add;
            client.request("abba.prepare", new JsonObject(), null, Duration.ofSeconds(5)); throw new AssertionError("Expected a refusal");
        } catch (IOException refused) {
            require(WorkerClient.reason(refused).equals("this job already ran"), "The worker's reason is read from its error: " + WorkerClient.reason(refused));
            require(!refused.getMessage().contains("secret-key"), "A worker's stderr never reaches an error message");
        }
        require(stderr.isEmpty(), "stderr of every request but abba.follow is drained only: " + stderr);
        long start = System.nanoTime();
        try (WorkerClient client = new WorkerClient(prefix)) {
            try { client.request("timeout", new JsonObject(), null, Duration.ofMillis(200)); throw new AssertionError("Expected timeout"); }
            catch (IOException expected) { require(expected.getMessage().contains("timed out"), "Timeout is actionable"); }
        }
        require((System.nanoTime() - start) < 5_000_000_000L, "Silent worker cannot block indefinitely");
        require(WorkerClient.reason(new IOException("{\"code\":\"runtime_error\",\"message\":\"Runtime request handling failed\"}"))
                .startsWith("{"), "An error without a reason keeps its text");
    }

    /** Records what the follower hands to ABBA; refuses the row "bad". */
    static final class Recorder implements Follower.Target, Follower.Output {
        final List<String> lines = Collections.synchronizedList(new ArrayList<>());
        final List<JsonArray> rows = Collections.synchronizedList(new ArrayList<>());
        final List<JsonObject> angles = Collections.synchronizedList(new ArrayList<>());
        public AbbaHostSession.ApplyReport apply(JsonArray updates, JsonObject hostAngles) {
            rows.add(updates); angles.add(hostAngles);
            AbbaHostSession.ApplyReport report = new AbbaHostSession.ApplyReport();
            for (JsonElement row : updates) {
                String id = row.getAsJsonObject().get("id").getAsString();
                if (id.equals("bad")) report.failed.put(id, "refused for the test"); else report.applied.add(id);
            }
            report.angles = hostAngles != null;
            return report;
        }
        public void line(String text) { lines.add(text); }
        public void status(String text) { lines.add("status: " + text); }
        void await(int applied) throws InterruptedException {
            long deadline = System.nanoTime() + 15_000_000_000L;
            while (rows.size() < applied && System.nanoTime() < deadline) Thread.sleep(20);
            require(rows.size() >= applied, "The follower applied " + rows.size() + " checkpoints, not " + applied);
        }
    }

    /** abba.follow: checkpoints in order, refusals logged and written to the jsonl file, the end on stdin's close. */
    static void followerChecks(Path prefix) throws Exception {
        Path root = Files.createTempDirectory("langslice follow ");
        Path job = Files.createDirectories(root.resolve("langslice"));
        Recorder recorder = new Recorder();
        Follower follower = new Follower(prefix, job, recorder, recorder);
        follower.start();
        recorder.await(2);
        require(recorder.rows.get(0).get(0).getAsJsonObject().get("id").getAsString().equals("a")
                && recorder.angles.get(0).get("pitch_deg").getAsDouble() == 2.0
                && recorder.rows.get(1).size() == 2 && recorder.angles.get(1) == null, "Both checkpoints applied, in order, with their angles");
        require(recorder.lines.contains("section a: warp sent"), "The worker's log lines reach Fiji's log: " + recorder.lines);
        require(recorder.lines.contains("LangSlice worker: follower warning"), "The follower's stderr reaches Fiji's log: " + recorder.lines);
        require(recorder.lines.stream().anyMatch(line -> line.startsWith("Not in ABBA yet") && line.contains("bad: refused for the test")),
                "A refused row is logged: " + recorder.lines);
        require(recorder.lines.stream().anyMatch(line -> line.startsWith("status: LangSlice: 2 steps applied")), "The status names the applied steps");
        List<String> applied = Files.readAllLines(job.resolve(Follower.APPLIED_LOG), StandardCharsets.UTF_8);
        JsonObject second = JsonParser.parseString(applied.get(1)).getAsJsonObject();
        require(applied.size() == 2 && second.has("at") && second.getAsJsonObject("failed").get("bad").getAsString().equals("refused for the test")
                && second.getAsJsonArray("applied").toString().equals("[\"b\"]") && !second.get("angles").getAsBoolean()
                && JsonParser.parseString(applied.get(0)).getAsJsonObject().get("angles").getAsBoolean(),
                "One jsonl line per applied checkpoint, with what failed and why: " + applied);
        require(follower.alive(), "The follower runs until it is stopped");
        long start = System.nanoTime();
        follower.stop();
        require(!follower.alive() && Files.isRegularFile(job.resolve("ended")), "Stopping closes stdin and the worker finishes by itself");
        require(System.nanoTime() - start < WorkerClient.GRACE.toNanos(), "A worker that answers stdin's end is not waited out");

        Path stubborn = Files.createDirectories(root.resolve("stubborn"));
        Recorder ignored = new Recorder();
        Follower forced = new Follower(prefix, stubborn, ignored, ignored);
        forced.start();
        ignored.await(2);
        forced.stop();
        require(!forced.alive() && !Files.exists(stubborn.resolve("ended")), "A worker that ignores stdin's end is terminated");

        Path waiting = Files.createDirectories(root.resolve("waiting"));
        Recorder idle = new Recorder();
        Follower prepared = new Follower(prefix, waiting, idle, idle);
        prepared.start();
        idle.await(1);
        prepared.stop();
        require(idle.rows.size() == 1 && idle.rows.get(0).size() == 0 && idle.angles.get(0) == null
                && idle.lines.stream().anyMatch(line -> line.contains("has not started yet")),
                "A prepared job that has not run: one empty application (the attach's removals) and a log line");
        require(!Files.exists(waiting.resolve(Follower.APPLIED_LOG)), "Nothing applied, nothing written");
        delete(root);
    }

    /** The window: `langslice gui JOB --tab TAB`, alive; one that ends at once is reported with its output. */
    static void windowChecks(Path prefix) throws Exception {
        Path folder = Files.createTempDirectory("langslice window ");
        Path job = folder.resolve("langslice");
        Process window = OpenLangSliceCommand.startWindow(prefix, folder, job, "prompt");
        try {
            require(window.isAlive(), "The window keeps running");
            List<String> log = OpenLangSliceCommand.firstLines(folder.resolve("gui.log"), 5);
            require(!log.isEmpty() && log.get(0).startsWith("argv ")
                    && JsonParser.parseString(log.get(0).substring(5)).equals(new Gson().toJsonTree(Arrays.asList("gui", job.toString(), "--tab", "prompt"))),
                    "The window is started as gui JOB --tab prompt: " + log);
        } finally { window.destroyForcibly(); }
        try {
            OpenLangSliceCommand.startWindow(prefix, folder, folder.resolve("fails"), "viewer");
            throw new AssertionError("A window that ends at once is an error");
        } catch (OpenLangSliceCommand.Stop expected) {
            require(expected.getMessage().contains("unrecognized arguments: --tab") && expected.getMessage().contains("Choose Python environment"),
                    "Its output and the way out are shown: " + expected.getMessage());
        }
        delete(folder);
    }

    /** The project key, finding a project folder again, candidates, continuing one, and the start-over move. */
    static void projectKeyChecks() throws Exception {
        String atlas = "Adult Mouse Brain - Allen Brain Atlas V3p1";
        String key = ProjectJob.key(atlas, Arrays.asList("s2", "s1", "s3"));
        require(key.matches("[0-9a-f]{12}") && key.equals(ProjectJob.key(atlas, Arrays.asList("s1", "s3", "s2"))),
                "The key is 12 hex digits and ignores slice order (and the selection, which it never reads)");
        require(!key.equals(ProjectJob.key(atlas, Arrays.asList("s1", "s2"))) && !key.equals(ProjectJob.key("other atlas", Arrays.asList("s1", "s2", "s3"))),
                "The key changes with the session's slices and atlas");
        require(AbbaHostSession.affineName(key).equals("LangSlice affine [" + key + "]") && AbbaHostSession.warpName(key).equals("LangSlice warp [" + key + "]"),
                "Step names carry the project key");
        Path root = Files.createTempDirectory("langslice projects ");
        ProjectJob.useRoot(root);
        try {
            require(ProjectJob.find(atlas, Arrays.asList("s1", "s2", "s3")).folder == null, "Nothing made yet");
            Path folder = root.resolve(key);
            JsonObject sections = new JsonObject(); sections.addProperty("section_0001.tif", "s1"); sections.addProperty("section_0002.tif", "s2");
            ProjectJob.create(folder, atlas, Arrays.asList("s3", "s1", "s2"), sections);
            Files.writeString(folder.resolve(AbbaHostSession.SECTIONS_FILE), sections.toString());
            require(folder.equals(ProjectJob.find(atlas, Arrays.asList("s3", "s2", "s1")).folder), "The same session finds its folder again");
            require(ProjectJob.sectionNames(folder).equals(new HashSet<>(Arrays.asList("s1", "s2"))), "The job's sections by ABBA slice name");
            require(!ProjectJob.hasJob(folder) && !ProjectJob.hasCheckpoint(folder), "No job.json or state.json yet");
            Files.createDirectories(ProjectJob.jobFolder(folder));
            Files.writeString(ProjectJob.jobFolder(folder).resolve("job.json"), "{}");
            require(ProjectJob.hasJob(folder) && !ProjectJob.hasCheckpoint(folder), "A prepared job has job.json and no checkpoint");
            Path other = root.resolve(ProjectJob.key("other atlas", Arrays.asList("s1")));
            ProjectJob.create(other, "other atlas", Arrays.asList("s1"), new JsonObject());
            ProjectJob.Found grown = ProjectJob.find(atlas, Arrays.asList("s1", "s2", "s3", "s4"));
            require(grown.folder == null && grown.candidates.size() == 1 && grown.candidates.get(0).folder.equals(folder)
                    && grown.candidates.get(0).shared == 3 && grown.candidates.get(0).sections == 2,
                    "A session with an added slice is offered the folder sharing its names (same atlas only): " + grown.candidates);
            ProjectJob.addKey(folder, grown.key);
            require(folder.equals(ProjectJob.find(atlas, Arrays.asList("s4", "s1", "s2", "s3")).folder), "Once continued, that session goes straight to it");
            require(ProjectJob.folderKey(folder).equals(key), "A continued folder keeps its own key for step names");
            Path aside = ProjectJob.startOver(folder);
            require(!Files.exists(folder) && Files.isRegularFile(aside.resolve(ProjectJob.PROJECT_FILE))
                    && aside.getFileName().toString().matches(key + "\\.old-\\d{8}-\\d{6}"), "Start over moves the folder aside: " + aside);
            ProjectJob.Found after = ProjectJob.find(atlas, Arrays.asList("s1", "s2", "s3"));
            require(after.folder == null && after.candidates.isEmpty(), "A folder set aside is never found or offered again");
            require(ProjectJob.medianMicrons(Arrays.asList(.3, .1, .2)) == 200 && ProjectJob.medianMicrons(Collections.emptyList()) == 0,
                    "Medians in µm; 0 when ABBA gives none");
        } finally { ProjectJob.useRoot(null); delete(root); }
    }

    /** Multi-page snapshots keep the channel order. */
    static void snapshotChecks() throws Exception {
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
        require(AbbaHostSession.exportedNames(Arrays.asList(0, 2), Arrays.asList("DAPI", "", "DAPI")).toString().equals("[\"DAPI\",\"DAPI (2)\"]")
                && AbbaHostSession.exportedNames(Arrays.asList(0, 1, 2), Arrays.asList("x", "x", "")).toString().equals("[\"x\",\"x (2)\",\"ch3\"]"),
                "channel_names follow the exported pages, never empty or repeated");
        delete(folder);
    }

    /** ABBA's Register menu gets one callable entry; Fiji's Plugins>LangSlice menu its three commands, once each. */
    @SuppressWarnings("unchecked")
    static void menuChecks() throws Exception {
        // Plugin annotations are compiled into SciJava's discovery index, not manually registered at runtime.
        java.io.InputStream index = LangSliceService.class.getResourceAsStream("/META-INF/json/org.scijava.plugin.Plugin");
        require(index != null, "SciJava plugin index exists");
        String plugins = new String(index.readAllBytes(), StandardCharsets.UTF_8); index.close();
        require(plugins.contains("OpenLangSliceCommand") && plugins.contains("StopFollowingCommand") && plugins.contains("SetupCommand")
                && !plugins.contains("RegistrationCommand") && !plugins.contains("AgentCommand"), "Plugin index lists only current commands");
        try (org.scijava.Context context = new org.scijava.Context(org.scijava.plugin.PluginService.class, org.scijava.command.CommandService.class, LangSliceService.class)) {
            org.scijava.command.CommandService commands = context.getService(org.scijava.command.CommandService.class);
            org.scijava.command.CommandInfo registration = commands.getCommand(LangSliceService.REGISTRATION);
            require(registration != null, "ABBA menu string resolves to a callable command");
            require(registration.createInstance() instanceof OpenLangSliceCommand, "Command alias constructs the open command");
            require(registration.getInput("mp") != null, "Alias preserves live-session parameter injection");
            java.lang.reflect.Field field = ch.epfl.biop.atlas.aligner.MultiSlicePositioner.class.getDeclaredField("externalRegistrationPluginsUI");
            field.setAccessible(true);
            Map<String, List<String>> ui = (Map<String, List<String>>) field.get(null);
            require(ui.get(LangSliceService.MENU).equals(Collections.singletonList(LangSliceService.REGISTRATION)), "One Register entry: " + ui);
            // ABBA builds the path as "Register>" + string.
            require(("Register>" + LangSliceService.REGISTRATION).equals("Register>LangSlice…"), "A plain Register entry, not a submenu");
            Map<String, Integer> entries = new TreeMap<>();
            for (org.scijava.command.CommandInfo info : commands.getCommands()) {
                org.scijava.MenuPath path = info.getMenuPath();
                if (path != null && !path.isEmpty() && path.getMenuString().startsWith("Plugins > LangSlice > ")) entries.merge(path.getMenuString(), 1, Integer::sum);
            }
            Map<String, Integer> expected = new TreeMap<>();
            for (String item : new String[]{"Open LangSlice…", "Choose Python environment…", "Stop following"}) expected.put("Plugins > LangSlice > " + item, 1);
            require(entries.equals(expected), "Plugins>LangSlice holds each command once: " + entries);
        }
    }
}
