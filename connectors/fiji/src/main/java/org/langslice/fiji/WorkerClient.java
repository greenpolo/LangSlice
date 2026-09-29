package org.langslice.fiji;

import com.google.gson.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

/** One request per isolated Python process. Never send credentials through command arguments. */
public final class WorkerClient implements AutoCloseable {
    private final Path environment;
    private volatile Process process;
    private volatile boolean closed;
    public WorkerClient(Path environment) { this.environment = environment; }
    public JsonObject request(String method, JsonObject params, Consumer<JsonObject> event, Duration timeout) throws Exception {
        if (closed) throw new IOException("LangSlice request cancelled");
        ProcessBuilder builder = new ProcessBuilder(EnvironmentDiscovery.command(environment));
        builder.environment().put("PYTHONUNBUFFERED", "1");
        builder.environment().remove("PYTHONPATH");
        builder.environment().remove("PYTHONHOME");
        Process child = builder.start();
        process = child;
        if (closed) { terminate(child); throw new IOException("LangSlice request cancelled"); }
        ExecutorService io = Executors.newFixedThreadPool(2, r -> { Thread t = new Thread(r, "LangSlice worker IO"); t.setDaemon(true); return t; });
        // Drain stderr to avoid deadlock; raw provider diagnostics may contain secrets and are not shown.
        Future<?> stderr = io.submit(() -> {
            try (InputStream stream = child.getErrorStream()) { byte[] bytes = new byte[4096]; while (stream.read(bytes) >= 0) { } }
            catch (IOException ignored) { }
        });
        String id = UUID.randomUUID().toString();
        Future<JsonObject> response = io.submit(() -> {
            try (BufferedReader input = new BufferedReader(new InputStreamReader(child.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = input.readLine()) != null) {
                    JsonObject message;
                    try { message = JsonParser.parseString(line).getAsJsonObject(); }
                    catch (RuntimeException malformed) { continue; } // Native libraries occasionally print to stdout.
                    if (!message.has("id") || message.get("id").isJsonNull() || !message.get("id").isJsonPrimitive() || !id.equals(message.get("id").getAsString())) continue;
                    String type = message.has("type") ? message.get("type").getAsString() : "";
                    if ("event".equals(type)) { if (event != null) event.accept(message.getAsJsonObject("event")); }
                    else if ("result".equals(type)) return message.getAsJsonObject("result");
                    else if ("error".equals(type)) {
                        // Authentication errors must not echo credentials or raw server replies.
                        if (method.startsWith("setup.")) throw new IOException("Setup could not complete. Check the installation, account details and internet connection, then try again.");
                        JsonElement error = message.get("error");
                        throw new IOException(error != null ? error.toString() : "LangSlice request failed");
                    }
                }
                throw new IOException("LangSlice did not return a result. Check the selected environment and installed version.");
            }
        });
        try {
            JsonObject request = new JsonObject(); request.addProperty("id", id); request.addProperty("method", method); request.add("params", params);
            try (Writer writer = new OutputStreamWriter(child.getOutputStream(), StandardCharsets.UTF_8)) { writer.write(request.toString()); writer.write('\n'); }
            return response.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException expired) { throw new IOException("LangSlice timed out. Check the environment or retry the operation.", expired); }
        catch (ExecutionException failed) { Throwable cause = failed.getCause(); if (cause instanceof Exception) throw (Exception) cause; throw failed; }
        finally { terminate(child); stderr.cancel(true); io.shutdownNow(); process = null; }
    }
    private static void terminate(Process child) {
        // conda run owns a Python child; cancelling only conda would leave it running.
        java.util.List<ProcessHandle> descendants = new java.util.ArrayList<>();
        child.descendants().forEach(descendants::add);
        java.util.Collections.reverse(descendants);
        descendants.forEach(handle -> { if (handle.isAlive()) handle.destroyForcibly(); });
        child.destroyForcibly();
    }
    @Override public void close() { closed = true; Process child = process; if (child != null) terminate(child); }
}
