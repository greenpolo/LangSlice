package org.langslice.fiji;

import com.google.gson.*;
import java.util.*;
import java.util.prefs.Preferences;

/** Every dialog choice as plain data: persisted between runs and turned into the worker's job spec. */
final class RegistrationSettings {
    static final String PROVIDER = "openai-oauth";
    static final String DEFAULT_MODEL = "openai-oauth/gpt-5.6-sol";
    static final String DEFAULT_IMAGE_MODEL = "gpt-image-2";
    static final List<String> FALLBACK_MODELS = Arrays.asList("openai-oauth/gpt-6-astra", "openai-oauth/gpt-6-sol",
            "openai-oauth/gpt-6-luna", "openai-oauth/gpt-5.6-sol", "openai-oauth/gpt-5.6-terra", "openai-oauth/gpt-5.6-luna");
    static final String[] REASONING = {"default", "low", "medium", "high", "xhigh", "max"};
    static final String[] LEVELS = {"low", "medium", "high"};
    static final Preferences PREFS = Preferences.userNodeForPackage(RegistrationSettings.class).node("registration");

    String model = DEFAULT_MODEL, imageModel = DEFAULT_IMAGE_MODEL, reasoning = "default", resolution = "low";
    boolean showLog = true, positioning = true, flip = true, linear = true, affine = true;
    boolean overwrite = false, agentDamage = true, custom = false, clahe = true, saveTraces = false;
    /** Where full agent traces go when saving is on. */
    String traceDir = java.nio.file.Paths.get(System.getProperty("user.home"), "LangSlice", "traces").toString();
    String cue = "", positionNotes = "", linearNotes = "", nonlinearNotes = "", strength = "medium";
    int thickness = 50, interval = 200, maxParallel = 4;
    double pixelSize = 25;
    double[] weights = {};
    /** No saved run yet: the dialog takes the worker's default models. */
    boolean fresh = true;

    static RegistrationSettings load(Preferences p) {
        RegistrationSettings s = new RegistrationSettings();
        s.fresh = p.get("model", null) == null;
        s.model = p.get("model", s.model); s.imageModel = p.get("imageModel", s.imageModel);
        s.reasoning = p.get("reasoning", s.reasoning); s.resolution = p.get("resolution", s.resolution);
        s.showLog = p.getBoolean("showLog", s.showLog); s.positioning = p.getBoolean("positioning", s.positioning);
        s.flip = p.getBoolean("flip", s.flip); s.linear = p.getBoolean("linear", s.linear); s.affine = p.getBoolean("affine", s.affine);
        s.overwrite = p.getBoolean("overwrite", s.overwrite); s.agentDamage = p.getBoolean("agentDamage", s.agentDamage);
        s.custom = p.getBoolean("custom", s.custom); s.clahe = p.getBoolean("clahe", s.clahe);
        s.cue = p.get("cue", s.cue); s.positionNotes = p.get("positionNotes", s.positionNotes);
        s.linearNotes = p.get("linearNotes", s.linearNotes); s.nonlinearNotes = p.get("nonlinearNotes", s.nonlinearNotes);
        s.strength = p.get("strength", s.strength);
        s.saveTraces = p.getBoolean("saveTraces", s.saveTraces); s.traceDir = p.get("traceDir", s.traceDir);
        s.thickness = p.getInt("thickness", s.thickness); s.interval = p.getInt("interval", s.interval);
        s.maxParallel = Math.max(1, Math.min(4, p.getInt("maxParallel", s.maxParallel)));
        s.pixelSize = p.getDouble("pixelSize", s.pixelSize);
        String saved = p.get("weights", "");
        if (!saved.isEmpty()) try { s.weights = Arrays.stream(saved.split(",")).mapToDouble(Double::parseDouble).toArray(); }
        catch (NumberFormatException ignored) { s.weights = new double[0]; }
        return s;
    }

    void save(Preferences p) {
        p.put("model", model); p.put("imageModel", imageModel); p.put("reasoning", reasoning); p.put("resolution", resolution);
        p.putBoolean("showLog", showLog); p.putBoolean("positioning", positioning); p.putBoolean("flip", flip);
        p.putBoolean("linear", linear); p.putBoolean("affine", affine); p.putBoolean("overwrite", overwrite);
        p.putBoolean("agentDamage", agentDamage); p.putBoolean("custom", custom); p.putBoolean("clahe", clahe);
        p.put("cue", cue); p.put("positionNotes", positionNotes); p.put("linearNotes", linearNotes);
        p.put("nonlinearNotes", nonlinearNotes); p.put("strength", strength);
        p.putBoolean("saveTraces", saveTraces); p.put("traceDir", traceDir);
        p.putInt("thickness", thickness); p.putInt("interval", interval); p.putInt("maxParallel", maxParallel);
        p.putDouble("pixelSize", pixelSize);
        StringBuilder text = new StringBuilder();
        for (double w : weights) text.append(text.length() == 0 ? "" : ",").append(w);
        p.put("weights", text.toString());
    }

    /** Saved weights are kept only when the channel count matches; otherwise every channel starts at 1. */
    double[] weightsFor(int channels) {
        if (weights.length == channels) return weights.clone();
        double[] fresh = new double[channels]; Arrays.fill(fresh, 1.0); return fresh;
    }

    static String modelId(String text) {
        String value = text == null ? "" : text.trim();
        return value.isEmpty() || value.contains("/") ? value : PROVIDER + "/" + value;
    }

    static String modelLabel(String id) { return id.startsWith(PROVIDER + "/") ? id.substring(PROVIDER.length() + 1) : id; }

    /** Auto exports every channel and lets the worker weigh them; custom exports only weighted channels. */
    List<Integer> exportChannels(int channels) {
        List<Integer> selected = new ArrayList<>();
        double[] w = weightsFor(channels);
        for (int c = 0; c < channels; c++) if (!custom || w[c] > 0) selected.add(c);
        return selected;
    }

    JsonObject preprocessing(int channels) {
        JsonObject value = new JsonObject();
        value.addProperty("mode", custom ? "custom" : "auto");
        if (!custom) return value;
        value.addProperty("clahe", clahe);
        value.addProperty("clahe_strength", strength);
        JsonArray list = new JsonArray();
        double[] w = weightsFor(channels);
        for (int c : exportChannels(channels)) list.add(w[c]);
        value.add("channel_weights", list);
        return value;
    }

    /** Null when runnable; otherwise a message for the user. */
    String problem(int channels) {
        if (!positioning && !linear) return "Turn on Positioning or Linear.";
        if (modelId(model).isEmpty()) return "Choose an agent model.";
        if (!(pixelSize > 0)) return "Choose a positive snapshot pixel size.";
        if (channels < 1) return "The selected slices have no image channel in common.";
        if (custom && exportChannels(channels).isEmpty()) return "Give at least one channel a weight above 0.";
        if (saveTraces && traceDir.trim().isEmpty()) return "Choose a folder for saved traces.";
        return null;
    }

    JsonObject spec() {
        JsonObject spec = new JsonObject();
        JsonArray tasks = new JsonArray();
        if (positioning) { tasks.add("reorder"); tasks.add("position"); }
        if (linear) tasks.add("transform");
        spec.add("tasks", tasks);
        spec.addProperty("model", modelId(model));
        if (!"default".equals(reasoning)) spec.addProperty("reasoning", reasoning);
        spec.addProperty("image_resolution", resolution);
        spec.addProperty("agent_damage", agentDamage);
        JsonObject order = new JsonObject();
        order.addProperty("flip", flip); order.addProperty("hemisphere_cue", cue.trim());
        spec.add("reorder", order);
        JsonObject position = new JsonObject();
        position.addProperty("thickness_um", thickness); position.addProperty("interval_um", interval);
        position.addProperty("notes", positionNotes.trim());
        spec.add("position", position);
        JsonObject transform = new JsonObject();
        transform.addProperty("automatic", affine); transform.addProperty("interactive", true);
        transform.addProperty("elastix", false); transform.addProperty("angles", false);
        transform.addProperty("max_parallel", maxParallel); transform.addProperty("notes", linearNotes.trim());
        spec.add("transform", transform);
        JsonObject nonlinear = new JsonObject();
        nonlinear.addProperty("provider", PROVIDER); nonlinear.addProperty("image_model", imageModel);
        nonlinear.addProperty("notes", nonlinearNotes.trim());
        spec.add("nonlinear", nonlinear);
        return spec;
    }

    /** Model lists from setup.status, or the connector's own list for older workers. */
    static List<String> models(JsonObject status, String key, List<String> fallback) {
        JsonObject account = account(status);
        List<String> values = new ArrayList<>();
        if (account != null && account.has(key) && account.get(key).isJsonArray())
            for (JsonElement item : account.getAsJsonArray(key)) values.add(item.getAsString());
        return values.isEmpty() ? new ArrayList<>(fallback) : values;
    }

    static String defaultValue(JsonObject status, String key, String fallback) {
        JsonObject account = account(status);
        return account != null && account.has(key) && account.get(key).isJsonPrimitive() ? account.get(key).getAsString() : fallback;
    }

    static boolean signedIn(JsonObject status) {
        JsonObject account = account(status);
        return account != null && account.has("configured") && account.get("configured").getAsBoolean();
    }

    private static JsonObject account(JsonObject status) {
        if (status == null || !status.has("providers") || !status.get("providers").isJsonObject()) return null;
        JsonElement account = status.getAsJsonObject("providers").get(PROVIDER);
        return account != null && account.isJsonObject() ? account.getAsJsonObject() : null;
    }
}
