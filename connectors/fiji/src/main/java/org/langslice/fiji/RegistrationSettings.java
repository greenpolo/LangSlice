package org.langslice.fiji;

import com.google.gson.*;
import java.util.*;
import java.util.prefs.Preferences;

/** Every dialog choice as plain data: persisted between runs and turned into the worker's job spec. */
final class RegistrationSettings {
    static final String PROVIDER = "openai-oauth";
    static final String DEFAULT_MODEL = "openai-oauth/gpt-5.6-sol";
    static final String DEFAULT_IMAGE_MODEL = "gpt-image-2";
    /** The image-model provider that runs Nonlinear without an image model: deformations fit the stain alone. */
    static final String NO_IMAGE_MODEL = "none";
    static final List<String> FALLBACK_MODELS = Arrays.asList("openai-oauth/gpt-6-astra", "openai-oauth/gpt-6-sol",
            "openai-oauth/gpt-6-luna", "openai-oauth/gpt-5.6-sol", "openai-oauth/gpt-5.6-terra", "openai-oauth/gpt-5.6-luna");
    static final String[] REASONING = {"default", "low", "medium", "high", "xhigh", "max"};
    static final String[] LEVELS = {"low", "medium", "high"};
    static final String[] RESOLUTIONS = {"low", "medium", "high", "auto"};
    /** nonlinear.engine values (core/spec.py DEFORMABLE_ENGINES), in the dialog's order. */
    static final String[] ENGINES = {"either", "ants", "elastix"};
    static final Preferences PREFS = Preferences.userNodeForPackage(RegistrationSettings.class).node("registration");

    boolean claude = false;
    String model = DEFAULT_MODEL, reasoning = "default", resolution = "low";
    /** The image model: its provider (canonical name, or "none") and model (null: the provider's default). */
    String imageProvider = PROVIDER, imageModel = DEFAULT_IMAGE_MODEL;
    boolean showLog = true, viewer = false, positioning = true, flip = true, linear = true, affine = true, angles = false;
    boolean nonlinear = false, overwrite = false, agentDamage = true, custom = false, clahe = true, saveTraces = false;
    boolean agentPreprocessing = false;
    /** Where full agent traces go when saving is on. */
    String traceDir = java.nio.file.Paths.get(System.getProperty("user.home"), "LangSlice", "traces").toString();
    String cue = "", positionNotes = "", linearNotes = "", nonlinearNotes = "", strength = "medium", engine = "either";
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
        s.engine = p.get("engine", s.engine);
        if (!Arrays.asList(ENGINES).contains(s.engine)) s.engine = "either";
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
        p.put("nonlinearNotes", nonlinearNotes); p.put("strength", strength); p.put("engine", engine);
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
        if (positioning) { tasks.add("reorder"); tasks.add("position"); }
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
        deform.addProperty("engine", engine);
        deform.addProperty("notes", nonlinearNotes.trim());
        spec.add("nonlinear", deform);
        return spec;
    }

    // ---- what setup.status offers ----------------------------------------------------------------

    /** One entry of the image-model list: a provider and a model (null = the provider's default). */
    static final class ImageChoice {
        final String provider, model, label;
        final boolean configured;
        ImageChoice(String provider, String model, String label, boolean configured) {
            this.provider = provider; this.model = model; this.label = label; this.configured = configured;
        }
        boolean matches(String provider, String model) {
            return this.provider.equals(provider) && Objects.equals(this.model == null ? "" : this.model, model == null ? "" : model);
        }
        @Override public String toString() { return label + (configured ? "" : " (not set up)"); }
    }

    static final String NONE_LABEL = "None (fit to the stain only)";
    private static final Map<String, String> PROVIDER_LABELS = new LinkedHashMap<>();
    static {
        PROVIDER_LABELS.put("openai-oauth", "ChatGPT");
        PROVIDER_LABELS.put("gemini-api", "Gemini API");
        PROVIDER_LABELS.put("openai-api", "OpenAI API");
    }

    /**
     * The image-model list, in the worker's order: its top-level "image_models" list from setup.status
     * ({provider, label, connected, models, default_model}; "image_providers" is read the same way), else each
     * provider entry's "image_models" from the older "providers" block, else (older workers) ChatGPT's image model.
     * The "none" provider is always offered, as "None (fit to the stain only)", last unless the worker places it.
     */
    static List<ImageChoice> imageChoices(JsonObject status) {
        List<ImageChoice> choices = new ArrayList<>();
        boolean none = false;
        JsonArray listed = null;
        for (String key : new String[]{"image_models", "image_providers"})
            if (listed == null && status != null && status.has(key) && status.get(key).isJsonArray()) listed = status.getAsJsonArray(key);
        if (listed != null) {
            for (JsonElement item : listed) {
                if (!item.isJsonObject()) continue;
                JsonObject entry = item.getAsJsonObject();
                String provider = text(entry, "provider", text(entry, "name", null));
                if (provider == null) continue;
                if (provider.equals(NO_IMAGE_MODEL)) { if (!none) choices.add(new ImageChoice(NO_IMAGE_MODEL, null, NONE_LABEL, true)); none = true; continue; }
                String label = text(entry, "label", PROVIDER_LABELS.getOrDefault(provider, provider));
                boolean configured = flag(entry, "connected", flag(entry, "configured", true));
                List<String> models = list(entry, "models");
                if (models.isEmpty()) models = list(entry, "image_models");
                addModels(choices, provider, label, configured, models,
                        text(entry, "default_model", text(entry, "default_image_model", null)));
            }
        } else if (status != null && status.has("providers") && status.get("providers").isJsonObject()) {
            JsonObject providers = status.getAsJsonObject("providers");
            List<String> order = new ArrayList<>(PROVIDER_LABELS.keySet());
            for (String key : providers.keySet()) if (!order.contains(key)) order.add(key);
            for (String provider : order) {
                if (provider.equals(NO_IMAGE_MODEL) || !providers.has(provider) || !providers.get(provider).isJsonObject()) continue;
                JsonObject entry = providers.getAsJsonObject(provider);
                List<String> models = list(entry, "image_models");
                if (models.isEmpty() && !PROVIDER_LABELS.containsKey(provider)) continue;
                addModels(choices, provider, PROVIDER_LABELS.getOrDefault(provider, provider), flag(entry, "configured", false), models,
                        text(entry, "default_image_model", null));
            }
        }
        if (choices.stream().noneMatch(choice -> !choice.provider.equals(NO_IMAGE_MODEL)))
            choices.add(0, new ImageChoice(PROVIDER, DEFAULT_IMAGE_MODEL, "ChatGPT: " + DEFAULT_IMAGE_MODEL, true));
        if (!none) choices.add(new ImageChoice(NO_IMAGE_MODEL, null, NONE_LABEL, true));
        return choices;
    }

    private static boolean flag(JsonObject entry, String key, boolean fallback) {
        return entry.has(key) && entry.get(key).isJsonPrimitive() ? entry.get(key).getAsBoolean() : fallback;
    }

    private static void addModels(List<ImageChoice> choices, String provider, String label, boolean configured,
            List<String> models, String fallback) {
        if (models.isEmpty() && fallback != null) models = Collections.singletonList(fallback);
        if (models.isEmpty()) choices.add(new ImageChoice(provider, null, label + ": default image model", configured));
        for (String model : models) choices.add(new ImageChoice(provider, model, label + ": " + model, configured));
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
