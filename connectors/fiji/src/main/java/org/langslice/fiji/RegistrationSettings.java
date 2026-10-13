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
    /** The tools the simple checkboxes switch off (spec.disabled names); the Advanced list mirrors them. */
    static final String MARK_DAMAGE = "mark_damage", ELASTIX_AFFINE = "elastix_affine", FLIP = "interactive_transform:flip";
    static final List<String> SIMPLE = Arrays.asList(MARK_DAMAGE, ELASTIX_AFFINE, FLIP);
    static final Preferences PREFS = Preferences.userNodeForPackage(RegistrationSettings.class).node("registration");
    /** The user's own thickness and interval; absent while the user has set none. */
    static final String USER_THICKNESS = "userThickness", USER_INTERVAL = "userInterval";

    // The job settings start from the worker's JobSpec defaults (setup.status "spec_defaults", the defaults
    // of langslice linear run, the MCP door and the library): see defaults(). The rest are the connector's own.
    boolean claude = false;
    /** The agent model; empty until the dialog fills in the worker's default. */
    String model = "", reasoning, resolution;
    /** The image model: its provider (canonical name, or "none") and model (null: the provider's default). */
    String imageProvider, imageModel;
    boolean showLog = true, viewer = false, positioning, flip, linear, affine, angles;
    boolean nonlinear, overwrite = false, agentDamage, custom = false, clahe = true;
    /** Tools and options switched off in the Advanced list, as spec.disabled names (SIMPLE names excluded: their checkboxes own them). */
    Set<String> disabled = new LinkedHashSet<>();
    String cue, positionNotes, linearNotes, nonlinearNotes, strength = "medium";
    /** The user's own section thickness and interval in µm (0 = infer); null: the user has set none. */
    Integer thickness, interval;
    /** ABBA's thickness and interval for the selected slices (0: none): shown and sent while the user has set none; never saved. */
    int abbaThickness, abbaInterval;
    int maxParallel;
    double pixelSize = 25;
    double[] weights = {};
    /** No saved run yet: the dialog takes the worker's default models. */
    boolean fresh = true;

    /** Settings that are the worker's JobSpec defaults (setup.status "spec_defaults"), nothing saved yet. */
    static RegistrationSettings defaults(JsonObject spec) {
        RegistrationSettings s = new RegistrationSettings();
        s.reasoning = spec.get("reasoning").isJsonNull() ? "default" : spec.get("reasoning").getAsString();
        s.resolution = spec.get("image_resolution").getAsString();
        List<String> tasks = new ArrayList<>();
        for (JsonElement task : spec.getAsJsonArray("tasks")) tasks.add(task.getAsString());
        s.positioning = tasks.contains("position"); s.linear = tasks.contains("transform"); s.nonlinear = tasks.contains("nonlinear");
        JsonObject position = spec.getAsJsonObject("position"), transform = spec.getAsJsonObject("transform");
        JsonObject deform = spec.getAsJsonObject("nonlinear");
        s.positionNotes = position.get("notes").getAsString();
        s.cue = transform.get("hemisphere_cue").getAsString(); s.angles = transform.get("angles").getAsBoolean();
        s.maxParallel = transform.get("max_parallel").getAsInt(); s.linearNotes = transform.get("notes").getAsString();
        s.imageProvider = deform.get("provider").getAsString();
        s.imageModel = deform.get("image_model").isJsonNull() ? null : deform.get("image_model").getAsString();
        s.nonlinearNotes = deform.get("notes").getAsString();
        Set<String> off = new LinkedHashSet<>();
        if (spec.has("disabled")) for (JsonElement name : spec.getAsJsonArray("disabled")) off.add(name.getAsString());
        s.agentDamage = !off.contains(MARK_DAMAGE); s.affine = !off.contains(ELASTIX_AFFINE); s.flip = !off.contains(FLIP);
        for (String name : off) if (!SIMPLE.contains(name)) s.disabled.add(name);
        return s;
    }

    /** The saved choices on top of the worker's JobSpec defaults (setup.status "spec_defaults"). */
    static RegistrationSettings load(Preferences p, JsonObject specDefaults) {
        RegistrationSettings s = defaults(specDefaults);
        s.claude = p.getBoolean("claude", false);
        s.fresh = p.get("model", null) == null;
        s.model = p.get("model", s.model);
        s.imageProvider = p.get("imageProvider", s.imageProvider);
        String image = p.get("imageModel", s.imageModel == null ? "" : s.imageModel);
        s.imageModel = image.isEmpty() ? null : image;
        s.reasoning = p.get("reasoning", s.reasoning); s.resolution = p.get("resolution", s.resolution);
        s.showLog = p.getBoolean("showLog", s.showLog); s.viewer = p.getBoolean("viewer", s.viewer);
        s.positioning = p.getBoolean("positioning", s.positioning);
        s.flip = p.getBoolean("flip", s.flip); s.linear = p.getBoolean("linear", s.linear); s.affine = p.getBoolean("affine", s.affine);
        s.angles = p.getBoolean("angles", s.angles); s.nonlinear = p.getBoolean("nonlinear", s.nonlinear);
        s.overwrite = p.getBoolean("overwrite", s.overwrite); s.agentDamage = p.getBoolean("agentDamage", s.agentDamage);
        s.custom = p.getBoolean("custom", s.custom); s.clahe = p.getBoolean("clahe", s.clahe);
        String off = p.get("disabled", null);
        if (off != null) { s.disabled.clear(); for (String name : off.split(",")) if (!name.isEmpty()) s.disabled.add(name); }
        s.cue = p.get("cue", s.cue); s.positionNotes = p.get("positionNotes", s.positionNotes);
        s.linearNotes = p.get("linearNotes", s.linearNotes); s.nonlinearNotes = p.get("nonlinearNotes", s.nonlinearNotes);
        s.strength = p.get("strength", s.strength);
        s.thickness = p.get(USER_THICKNESS, null) == null ? null : p.getInt(USER_THICKNESS, 0);
        s.interval = p.get(USER_INTERVAL, null) == null ? null : p.getInt(USER_INTERVAL, 0);
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
        p.put("disabled", String.join(",", disabled));
        p.put("cue", cue); p.put("positionNotes", positionNotes); p.put("linearNotes", linearNotes);
        p.put("nonlinearNotes", nonlinearNotes); p.put("strength", strength);
        if (thickness == null) p.remove(USER_THICKNESS); else p.putInt(USER_THICKNESS, thickness);
        if (interval == null) p.remove(USER_INTERVAL); else p.putInt(USER_INTERVAL, interval);
        p.putInt("maxParallel", maxParallel);
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

    /** Auto exports every channel and lets the worker weigh them; Custom exports only weighted channels. */
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

    /**
     * spec.disabled: the simple checkboxes' tools (damage marks, the affine tool, the mirror) first, then the
     * Advanced list's names, each once.
     */
    JsonArray disabledNames() {
        Set<String> names = new LinkedHashSet<>();
        if (!agentDamage) names.add(MARK_DAMAGE);
        if (!affine) names.add(ELASTIX_AFFINE);
        if (!flip) names.add(FLIP);
        for (String name : disabled) if (!SIMPLE.contains(name)) names.add(name);
        JsonArray list = new JsonArray();
        for (String name : names) list.add(name);
        return list;
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
        spec.add("disabled", disabledNames());
        JsonObject position = new JsonObject();
        int thicknessUm = thickness != null ? thickness : abbaThickness, intervalUm = interval != null ? interval : abbaInterval;
        if (thicknessUm > 0) position.addProperty("thickness_um", thicknessUm);
        if (intervalUm > 0) position.addProperty("interval_um", intervalUm);
        position.addProperty("notes", positionNotes.trim());
        spec.add("position", position);
        JsonObject transform = new JsonObject();
        transform.addProperty("hemisphere_cue", cue.trim());
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

    /** One entry of setup.status "tools": a tool (when disableable) and its options ("tool:option" names). */
    static final class Tool {
        final String name, group;
        final boolean disableable;
        final List<String> options;
        /** What the tool needs while on: groups of names, one of each group on. */
        final List<List<String>> needs;
        Tool(String name, String group, boolean disableable, List<String> options, List<List<String>> needs) {
            this.name = name; this.group = group; this.disableable = disableable; this.options = options; this.needs = needs;
        }
    }

    /** The tool catalog from setup.status ("tools", from langslice.core.toolset.catalog()). */
    static List<Tool> toolCatalog(JsonObject status) {
        List<Tool> tools = new ArrayList<>();
        for (JsonElement item : status.getAsJsonArray("tools")) {
            JsonObject entry = item.getAsJsonObject();
            List<List<String>> needs = new ArrayList<>();
            for (JsonElement alternatives : entry.getAsJsonArray("needs")) {
                List<String> names = new ArrayList<>();
                for (JsonElement n : alternatives.getAsJsonArray()) names.add(n.getAsString());
                needs.add(names);
            }
            List<String> options = new ArrayList<>();
            for (JsonElement option : entry.getAsJsonArray("options")) options.add(option.getAsString());
            tools.add(new Tool(entry.get("name").getAsString(), entry.get("group").getAsString(),
                    entry.get("disableable").getAsBoolean(), options, needs));
        }
        return tools;
    }

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
        for (JsonElement item : status.getAsJsonArray("image_models")) {
            JsonObject entry = item.getAsJsonObject();
            String provider = entry.get("provider").getAsString();
            if (provider.equals(NO_IMAGE_MODEL)) { choices.add(new ImageChoice(NO_IMAGE_MODEL, null, NONE_LABEL, true)); continue; }
            String label = entry.get("label").getAsString();
            boolean connected = entry.get("connected").getAsBoolean();
            for (JsonElement model : entry.getAsJsonArray("models"))
                choices.add(new ImageChoice(provider, model.getAsString(), label + ": " + model.getAsString(), connected));
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
