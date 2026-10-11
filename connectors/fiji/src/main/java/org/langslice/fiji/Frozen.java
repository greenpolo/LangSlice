package org.langslice.fiji;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

/**
 * The spec's {@code frozen} map (see core/frozen.py): section name to the properties the agent may not change.
 * A registered slice whose registrations are kept freezes its orientation and affine; a slice the user warped
 * also keeps its deformation then; a slice left out of Nonlinear freezes its deformation.
 */
final class Frozen {
    static final String ORIENTATION = "orientation", AFFINE = "affine", DEFORMATION = "deformation";

    private Frozen() { }

    /** Adds properties to one section's entry of spec.frozen, each at most once. */
    static void add(JsonObject spec, String section, String... properties) {
        if (!spec.has("frozen")) spec.add("frozen", new JsonObject());
        JsonObject map = spec.getAsJsonObject("frozen");
        JsonArray held = map.has(section) ? map.getAsJsonArray(section) : new JsonArray();
        for (String property : properties) {
            boolean present = false;
            for (JsonElement element : held) present |= element.getAsString().equals(property);
            if (!present) held.add(property);
        }
        map.add(section, held);
    }

    /** A registered section whose registrations are kept; {@code keepWarp} adds its deformation too. */
    static void registered(JsonObject spec, String section, boolean keepWarp) {
        add(spec, section, ORIENTATION, AFFINE);
        if (keepWarp) add(spec, section, DEFORMATION);
    }
}
