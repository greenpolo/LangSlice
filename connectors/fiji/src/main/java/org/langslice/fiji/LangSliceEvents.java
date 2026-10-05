package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.SliceSources;
import com.google.gson.JsonObject;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

/**
 * Where an in-JVM listener follows LangSlice runs: a Python proxy registered through JPype when ABBA was
 * started by {@code langslice abba}, which drives LangSlice's agent viewer. Nothing here needs Python.
 *
 * <p>Each message is one JSON object (as text) with a {@code kind}. The worker's own messages are passed on
 * verbatim, in both modes: {@code agent_event} ({@code event}: the agent's tool_start / tool_end / seed ...),
 * {@code checkpoint} (host_updates, host_angles, updates_since_start, state) and {@code log} (message). The
 * connector adds {@code run_started} (mode "openai-oauth" or "claude", viewer: the user ticked "Open agent viewer",
 * image_folder, sections: snapshot filename to ABBA slice name), {@code job} (Claude mode: job_id, job_dir),
 * {@code applied} (what reached ABBA: applied ids, failed id to reason, angles) and {@code run_finished}
 * (message, and job_dir when known). Delivery runs on one background thread, in order; a failing listener is
 * logged and never affects the run or the other listeners.
 */
public final class LangSliceEvents {
    private static final List<Consumer<String>> LISTENERS = new CopyOnWriteArrayList<>();
    private static final ExecutorService DELIVERY = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "LangSlice event delivery");
        thread.setDaemon(true);
        return thread;
    });
    private static volatile Map<String, SliceSources> slices = Collections.emptyMap();

    private LangSliceEvents() { }

    /** Registers a listener; it receives every later message as JSON text. */
    public static void addListener(Consumer<String> listener) { LISTENERS.add(Objects.requireNonNull(listener)); }

    public static boolean removeListener(Consumer<String> listener) { return LISTENERS.remove(listener); }

    /** True when a listener (normally LangSlice's Python launcher) is registered: the agent viewer is available. */
    public static boolean hasListeners() { return !LISTENERS.isEmpty(); }

    /** The current (or last) run's snapshot filename to ABBA slice, in snapshot order. */
    public static Map<String, SliceSources> slices() { return slices; }

    static void runStarted(Map<String, SliceSources> sections, JsonObject message) {
        slices = Collections.unmodifiableMap(new LinkedHashMap<>(sections));
        JsonObject names = new JsonObject();
        sections.forEach((file, slice) -> names.addProperty(file, slice.getName()));
        message.add("sections", names);
        publish("run_started", message);
    }

    /** Sends one message of the given kind; the body's keys are kept. Returns at once. */
    static void publish(String kind, JsonObject body) {
        if (LISTENERS.isEmpty()) return;
        JsonObject message = body == null ? new JsonObject() : body.deepCopy();
        message.addProperty("kind", kind);
        send(message.toString());
    }

    /** Passes one of the worker's own messages on unchanged (it carries its own kind). */
    static void forward(JsonObject payload) {
        if (!LISTENERS.isEmpty()) send(payload.toString());
    }

    private static void send(String text) {
        DELIVERY.execute(() -> {
            for (Consumer<String> listener : LISTENERS) {
                try { listener.accept(text); }
                catch (Throwable failure) { System.err.println("LangSlice event listener failed: " + failure); }
            }
        });
    }

    /** Waits until every message sent so far was delivered (tests). */
    static void flush() throws Exception { DELIVERY.submit(() -> { }).get(10, TimeUnit.SECONDS); }
}
