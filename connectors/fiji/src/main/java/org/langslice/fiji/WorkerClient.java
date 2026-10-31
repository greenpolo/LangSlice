package org.langslice.fiji;

import com.google.gson.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

/**
 * One request per isolated Python process ({@code langslice serve --stdio}). The request line is written and stdin
 * stays open until the result: a long request (the follower, {@code abba.follow}) ends when {@link #close} closes
 * stdin, and its process is terminated if it does not end by itself. Never send credentials through arguments.
 */
public final class WorkerClient implements AutoCloseable {
    /** How long {@link #close} lets the worker answer stdin's end before terminating it. */
    static final Duration GRACE = Duration.ofSeconds(3);
    private final Path environment;
    private volatile Process process;
    private volatile Writer stdin;
    private volatile boolean closed;
    /** Where abba.follow's stderr lines go (Fiji's log); a test replaces it. */
    volatile Consumer<String> stderrLog = FijiLog::line;

    public WorkerClient(Path environment) { this.environment = environment; }

    /**
     * Sends one request and returns its result; *event* receives each streamed event's object in order.
     * A null *timeout* waits for as long as the request runs (the follower).
     */
    public JsonObject request(String method, JsonObject params, Consumer<JsonObject> event, Duration timeout) throws Exception {
        if (closed) throw new IOException("LangSlice request cancelled");
        ProcessBuilder builder = new ProcessBuilder(EnvironmentDiscovery.command(environment, "serve", "--stdio"));
        clean(builder);
        Process child = builder.start();
        process = child;
        if (closed) { terminate(child); throw new IOException("LangSlice request cancelled"); }
        ExecutorService io = Executors.newFixedThreadPool(2, r -> { Thread t = new Thread(r, "LangSlice worker IO"); t.setDaemon(true); return t; });
        // The follower's warnings and tracebacks go to Fiji's log. Every other request's stderr is drained only.
        boolean forward = method.equals("abba.follow");
        Future<?> stderr = io.submit(() -> {
            try (BufferedReader lines = new BufferedReader(new InputStreamReader(child.getErrorStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = lines.readLine()) != null) if (forward && !line.isBlank()) stderrLog.accept("LangSlice worker: " + line);
            } catch (IOException ignored) { }
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
                    if ("event".equals(type)) { if (event != null && message.get("event").isJsonObject()) event.accept(message.getAsJsonObject("event")); }
                    else if ("result".equals(type)) return message.getAsJsonObject("result");
                    else if ("error".equals(type)) {
                        JsonElement error = message.get("error");
                        throw new IOException(error != null ? error.toString() : "LangSlice request failed");
                    }
                }
                throw new IOException("LangSlice did not return a result. Check the selected environment and installed version.");
            }
        });
        try {
            JsonObject request = new JsonObject(); request.addProperty("id", id); request.addProperty("method", method); request.add("params", params);
            Writer writer = new OutputStreamWriter(child.getOutputStream(), StandardCharsets.UTF_8);
            stdin = writer;
            writer.write(request.toString()); writer.write('\n'); writer.flush();
            if (closed) closeStdin();
            return timeout == null ? response.get() : response.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException expired) { throw new IOException("LangSlice timed out. Check the environment or retry the operation.", expired); }
        catch (ExecutionException failed) { Throwable cause = failed.getCause(); if (cause instanceof Exception) throw (Exception) cause; throw failed; }
        finally { closeStdin(); terminate(child); stderr.cancel(true); io.shutdownNow(); process = null; }
    }

    /**
     * Starts {@code langslice ARGS} in *prefix* as a process of its own (the LangSlice window), its output appended to
     * *log*. It is not ended with Fiji or ABBA.
     */
    public static Process start(Path prefix, Path log, String... args) throws IOException {
        ProcessBuilder builder = new ProcessBuilder(EnvironmentDiscovery.command(prefix, args));
        clean(builder);
        builder.redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.appendTo(log.toFile()));
        builder.redirectInput(ProcessBuilder.Redirect.from(new File(System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win") ? "NUL" : "/dev/null")));
        return builder.start();
    }

    private static void clean(ProcessBuilder builder) {
        builder.environment().put("PYTHONUNBUFFERED", "1");
        builder.environment().remove("PYTHONPATH");
        builder.environment().remove("PYTHONHOME");
    }

    private void closeStdin() {
        Writer writer = stdin;
        stdin = null;
        if (writer != null) try { writer.close(); } catch (IOException ignored) { }
    }

    private static void terminate(Process child) {
        // conda run owns a Python child; cancelling only conda would leave it running.
        List<ProcessHandle> descendants = new ArrayList<>();
        child.descendants().forEach(descendants::add);
        Collections.reverse(descendants);
        descendants.forEach(handle -> { if (handle.isAlive()) handle.destroyForcibly(); });
        child.destroyForcibly();
    }

    /** Ends the request: stdin is closed (the follower's signal to finish), then the worker is terminated after {@link #GRACE}. */
    @Override public void close() {
        closed = true;
        closeStdin();
        Process child = process;
        if (child == null) return;
        try { if (child.waitFor(GRACE.toMillis(), TimeUnit.MILLISECONDS)) return; }
        catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        terminate(child);
    }

    /** A worker error's plain reason ({code, message, details: {error}}), else the exception's message. */
    static String reason(Throwable error) {
        String reason = workerReason(error);
        return reason == null || reason.isEmpty() ? String.valueOf(error.getMessage()) : reason;
    }

    /** The reason inside a worker error ({code, message, details: {error}}), or null. */
    static String workerReason(Throwable error) {
        String text = error == null ? null : error.getMessage();
        if (text == null) return null;
        try {
            JsonObject parsed = JsonParser.parseString(text).getAsJsonObject();
            if (parsed.has("details") && parsed.get("details").isJsonObject() && parsed.getAsJsonObject("details").has("error"))
                return parsed.getAsJsonObject("details").get("error").getAsString().trim();
            String message = parsed.has("message") ? parsed.get("message").getAsString() : null;
            return message == null || message.equals("Runtime request handling failed") ? null : message.trim();
        } catch (RuntimeException notJson) { return null; }
    }
}
