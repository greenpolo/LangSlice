package org.langslice.fiji;

import com.google.gson.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

/** Offline executable integration checks, using a real fake-worker subprocess. */
public final class ConnectorSmokeTest {
    static void require(boolean condition, String message) { if (!condition) throw new AssertionError(message); }
    public static void main(String[] args) throws Exception {
        Path prefix = Files.createTempDirectory("langslice connector space ");
        Path python = prefix.resolve("bin/python"); Files.createDirectories(python.getParent());
        String script = "#!/usr/bin/env python3\nimport json,sys,time,subprocess\nr=json.loads(sys.stdin.readline())\n" +
            "if r['method']=='timeout':\n time.sleep(20)\n" +
            "elif r['method']=='setup.fail':\n print(json.dumps({'id':r['id'],'type':'error','error':'secret-key-must-not-escape'}))\n" +
            "else:\n print(json.dumps({'id':None,'type':'event'}))\n print(json.dumps({'id':r['id'],'type':'event','event':{'kind':'data','payload':{'kind':'checkpoint'}}}))\n print(json.dumps({'id':r['id'],'type':'result','result':{'protocol_version':1}}))\n";
        Files.write(python, script.getBytes(StandardCharsets.UTF_8)); python.toFile().setExecutable(true);
        require(EnvironmentDiscovery.python(prefix).equals(python), "Prefix resolves python");
        require(EnvironmentDiscovery.command(prefix).get(0).equals(python.toString()), "Prefix with spaces stays a single argv element");
        EnvironmentDiscovery.save(prefix);
        require(prefix.equals(EnvironmentDiscovery.saved()), "Remember environment path with spaces");
        require(EnvironmentDiscovery.discover().contains(prefix), "Saved environment is rediscovered");
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
        // Plugin annotations are compiled into SciJava's discovery index, not manually registered at runtime.
        java.io.InputStream index = LangSliceService.class.getResourceAsStream("/META-INF/json/org.scijava.plugin.Plugin");
        require(index != null, "SciJava plugin index exists"); index.close();
        try (org.scijava.Context context = new org.scijava.Context(org.scijava.plugin.PluginService.class, org.scijava.command.CommandService.class, LangSliceService.class)) {
            org.scijava.command.CommandService commands = context.getService(org.scijava.command.CommandService.class);
            org.scijava.command.CommandInfo setup = commands.getCommand(LangSliceService.SETUP);
            require(setup != null, "ABBA menu string resolves to a callable command");
            require(setup.createInstance() instanceof SetupCommand, "Command alias constructs native setup command");
            require(setup.getInput("mp") != null, "Alias preserves live-session parameter injection");
            require(commands.getCommand(LangSliceService.AGENT) != null, "Agent alias is discoverable");
        }
        Files.delete(python); Files.delete(python.getParent()); Files.delete(prefix);
        System.out.println("Connector smoke checks passed.");
    }
}
