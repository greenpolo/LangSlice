package org.langslice.fiji;

import com.google.gson.*;
import java.util.*;
import java.util.prefs.Preferences;

/** Every dialog choice as plain data: persisted between runs and turned into the worker's job spec. */
final class RegistrationSettings {
    /** The ChatGPT account that runs the agent in ChatGPT mode (providers/registry.py). */
    static final String PROVIDER = "openai-oauth";
    /** The image-model provider that runs Nonlinear without an image model: deformations fit the stain alone. */
    static final String NO_IMAGE_MODEL = "none";
    static final String[] REASONING = {"default", "low", "medium", "high", "xhigh", "max"};
    static final String[] LEVELS = {"low", "medium", "high"};
    static final String[] RESOLUTIONS = {"low", "medium", "high", "auto"};
    static final Preferences PREFS = Preferences.userNodeForPackage(RegistrationSettings.class).node("registration");

    boolean claude = false;
    /** The agent model; empty until the dialog fills in the worker's default. */
    String model = "", reasoning = "default", resolution = "low";
    /** The image model: its provider (canonical name, or "none") and model (null: the provider's default). */
    String imageProvider = PROVIDER, imageModel = null;
    boolean showLog = true, viewer = false, positioning = true, flip = true, linear = true, affine = true, angles = false;
    boolean nonlinear = false, overwrite = false, agentDamage = true, custom = false, clahe = true, saveTraces = false;
    boolean agentPreprocessing = false;
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
        s.claude = p.getBoolean("claude", false);
        s.fresh = p.get("model", null) == null;
        s.model = p.get("model", s.model);
        s.imageProvider = p.get("imageProvider", s.imageProvider);
        String image = p.get("imageModel", s.imageModel);
        s.imageModel = image.isEmpty() ? null : image;
        s.reasoning = p.get("reasoning", s.reasoning); s.resolution = p.get("resolution", s.resolution);
        s.showLog = p.getBoolean("showLog", s.showLog); s.viewer = p.getBoolean("viewer", s.viewer);
        s.positioning = p.getBoolean("positioning", s.positioning);
        s.flip = p.getBoolean("flip", s.flip); s.linear = p.getBoolean("linear", s.linear); s.affine = p.getBoolean("affine", s.affine);
        s.angles = p.getBoolean("angles", s.angles); s.nonlinear = p.getBoolean("nonlinear", s.nonlinear);
        s.overwrite = p.getBoolean("overwrite", s.overwrite); s.agentDamage = p.getBoolean("agentDamage", s.agentDamage);
        s.custom = p.getBoolean("custom", s.custom); s.clahe = p.getBoolean("clahe", s.clahe);
        s.agentPreprocessing = p.getBoolean("agentPreprocessing", s.agentPreprocessing);
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
        p.putBoolean("claude", claude);
        p.put("model", model); p.put("imageProvider", imageProvider); p.put("imageModel", imageModel == null ? "" : imageModel);
        p.put("reasoning", reasoning); p.put("resolution", resolution);
        p.putBoolean("showLog", showLog); p.putBoolean("viewer", viewer); p.putBoolean("positioning", positioning); p.putBoolean("flip", flip);
        p.putBoolean("linear", linear); p.putBoolean("affine", affine); p.putBoolean("angles", angles); p.putBoolean("nonlinear", nonlinear);
        p.putBoolean("overwrite", overwrite);
        p.putBoolean("agentDamage", agentDamage); p.putBoolean("custom", custom); p.putBoolean("clahe", clahe);
        p.putBoolean("agentPreprocessing", agentPreprocessing);
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

    /**
     * Auto exports every channel and lets the worker weigh them; Custom exports only weighted channels,
     * unless the agent drives preprocessing: then every channel is exported, so the agent can use all of them.
     */
    List<Integer> exportChannels(int channels) {
        List<Integer> selected = new ArrayList<>();
        double[] w = weightsFor(channels);
        for (int c = 0; c < channels; c++) if (!custom || agentPreprocessing || w[c] > 0) selected.add(c);
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

    boolean usesImageModel() { return nonlinear && !NO_IMAGE_MODEL.equals(imageProvider); }

    /** Null when runnable; otherwise a message for the user. */
    String problem(int channels) {
        if (!positioning && !linear && !nonlinear) return "Turn on Positioning, Linear or Nonlinear.";
        if (!claude && modelId(model).isEmpty()) return "Choose an agent model.";
        if (!(pixelSize > 0)) return "Choose a positive snapshot pixel size.";
        if (channels < 1) return "The selected slices have no image channel in common.";
        if (custom && exportChannels(channels).stream().noneMatch(c -> weightsFor(channels)[c] > 0))
            return "Give at least one channel a weight above 0.";
        if (saveTraces && traceDir.trim().isEmpty()) return "Choose a folder for saved traces.";
        return null;
    }

    JsonObject spec() {
        JsonObject spec = new JsonObject();
        JsonArray tasks = new JsonArray();
        if (positioning) { tasks.add("position"); }
        if (linear) tasks.add("transform");
        if (nonlinear) tasks.add("nonlinear");
        spec.add("tasks", tasks);
        spec.addProperty("model", modelId(model));
        if (!"default".equals(reasoning)) spec.addProperty("reasoning", reasoning);
        spec.addProperty("image_resolution", resolution);
        spec.addProperty("agent_damage", agentDamage);
        spec.addProperty("agent_preprocessing", agentPreprocessing);
        JsonObject position = new JsonObject();
        position.addProperty("thickness_um", thickness); position.addProperty("interval_um", interval);
        position.addProperty("notes", positionNotes.trim());
        spec.add("position", position);
        JsonObject transform = new JsonObject();
        transform.addProperty("flip", flip); transform.addProperty("hemisphere_cue", cue.trim());
        transform.addProperty("automatic", affine); transform.addProperty("interactive", true);
        transform.addProperty("angles", angles);
        transform.addProperty("max_parallel", maxParallel); transform.addProperty("notes", linearNotes.trim());
        spec.add("transform", transform);
        JsonObject deform = new JsonObject();
        deform.addProperty("provider", imageProvider);
        if (NO_IMAGE_MODEL.equals(imageProvider) || imageModel == null || imageModel.isEmpty()) deform.add("image_model", JsonNull.INSTANCE);
        else deform.addProperty("image_model", imageModel);
        deform.addProperty("notes", nonlinearNotes.trim());
        spec.add("nonlinear", deform);
        return spec;
    }

    // ---- what setup.status offers ----------------------------------------------------------------

    /** One entry of the image-model list: a provider and a model (null = the provider's default). */
    static final class ImageChoice {
        final String provider, model, label;
        final boolean connected;
        ImageChoice(String provider, String model, String label, boolean connected) {
            this.provider = provider; this.model = model; this.label = label; this.connected = connected;
        }
        boolean matches(String provider, String model) {
            return this.provider.equals(provider) && Objects.equals(this.model == null ? "" : this.model, model == null ? "" : model);
        }
        @Override public String toString() { return label + (connected ? "" : " (not set up)"); }
    }

    static final String NONE_LABEL = "None (fit to the stain only)";

    /**
     * The image-model list, in the worker's order: setup.status's "image_models" ({provider, label, connected,
     * models, default_model}), one choice per model; the "none" provider reads "None (fit to the stain only)".
     */
    static List<ImageChoice> imageChoices(JsonObject status) {
        List<ImageChoice> choices = new ArrayList<>();
        if (status == null || !status.has("image_models") || !status.get("image_models").isJsonArray()) return choices;
        for (JsonElement item : status.getAsJsonArray("image_models")) {
            if (!item.isJsonObject()) continue;
            JsonObject entry = item.getAsJsonObject();
            String provider = text(entry, "provider", null);
            if (provider == null) continue;
            if (provider.equals(NO_IMAGE_MODEL)) { choices.add(new ImageChoice(NO_IMAGE_MODEL, null, NONE_LABEL, true)); continue; }
            String label = text(entry, "label", provider);
            boolean connected = entry.has("connected") && entry.get("connected").isJsonPrimitive() && entry.get("connected").getAsBoolean();
            List<String> models = list(entry, "models");
            String fallback = text(entry, "default_model", null);
            if (models.isEmpty() && fallback != null) models = Collections.singletonList(fallback);
            if (models.isEmpty()) choices.add(new ImageChoice(provider, null, label + ": default image model", connected));
            for (String model : models) choices.add(new ImageChoice(provider, model, label + ": " + model, connected));
        }
        return choices;
    }

    private static String text(JsonObject entry, String key, String fallback) {
        return entry.has(key) && entry.get(key).isJsonPrimitive() ? entry.get(key).getAsString() : fallback;
    }

    private static List<String> list(JsonObject entry, String key) {
        List<String> values = new ArrayList<>();
        if (entry.has(key) && entry.get(key).isJsonArray())
            for (JsonElement item : entry.getAsJsonArray(key)) if (item.isJsonPrimitive()) values.add(item.getAsString());
        return values;
    }

    /** The ChatGPT account's agent models from setup.status (providers.openai-oauth.agent_models). */
    static List<String> agentModels(JsonObject status) {
        JsonObject account = account(status);
        return account == null ? new ArrayList<>() : list(account, "agent_models");
    }

    /** One of the ChatGPT account's defaults (default_agent_model, default_image_model), or null. */
    static String accountDefault(JsonObject status, String key) {
        JsonObject account = account(status);
        return account == null ? null : text(account, key, null);
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
