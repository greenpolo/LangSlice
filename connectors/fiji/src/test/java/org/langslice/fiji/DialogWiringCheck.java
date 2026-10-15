package org.langslice.fiji;

import com.google.gson.*;
import java.awt.image.BufferedImage;
import java.lang.reflect.Field;
import java.util.*;
import javax.swing.*;

/**
 * Optional, needs a display and no ABBA: DialogWiringCheck. Checks the Registration dialog's Advanced list
 * inside the real dialog: the simple checkboxes mirror their list entries, catalog checkboxes reach spec.disabled,
 * and the choices persist.
 */
public final class DialogWiringCheck {
    static void require(boolean ok, String message) { if (!ok) throw new AssertionError(message); }

    static final String TOOLS = "\"tools\":[{\"name\":\"look\",\"group\":\"Common\",\"kind\":\"read\",\"disableable\":false,\"options\":[\"section\",\"atlas\"],\"needs\":[]},"
            + "{\"name\":\"zoom\",\"group\":\"Common\",\"kind\":\"read\",\"disableable\":true,\"options\":[],\"needs\":[]},"
            + "{\"name\":\"mark_damage\",\"group\":\"Common\",\"kind\":\"write\",\"disableable\":true,\"options\":[],\"needs\":[]},"
            + "{\"name\":\"interactive_transform\",\"group\":\"Linear\",\"kind\":\"write\",\"disableable\":true,\"options\":[\"flip\"],\"needs\":[[\"look:overlay\"]]},"
            + "{\"name\":\"elastix_affine\",\"group\":\"Linear\",\"kind\":\"write\",\"disableable\":true,\"options\":[],\"needs\":[]},"
            + "{\"name\":\"fireants_syn\",\"group\":\"Nonlinear\",\"kind\":\"write\",\"disableable\":true,\"options\":[],\"needs\":[]}]";

    static JsonObject status() {
        String text = "{\"providers\":{\"openai-oauth\":{\"configured\":true,\"agent_models\":[\"openai-oauth/gpt-6-sol\"],"
                + "\"default_agent_model\":\"openai-oauth/gpt-6-sol\"}}," + TOOLS + "}";
        return JsonParser.parseString(text).getAsJsonObject();
    }

    static RegistrationDialog dialog(JsonObject status, RegistrationSettings settings) throws Exception {
        List<RegistrationDialog.SliceRow> rows = Collections.singletonList(new RegistrationDialog.SliceRow("s1.tif", 0));
        RegistrationDialog.Host host = new RegistrationDialog.Host() {
            public BufferedImage[] preview(int slice, List<Integer> channels, JsonObject p, double px) { return null; }
            public JsonObject estimate(JsonObject params) { return null; }
            public JsonObject status() { return status; }
            public void setup() { }
            public void run(RegistrationSettings s, Map<Integer, String> d, Set<Integer> n) { }
        };
        RegistrationDialog[] out = new RegistrationDialog[1];
        SwingUtilities.invokeAndWait(() -> out[0] = new RegistrationDialog(status, rows, "x", Collections.singletonList("DAPI"), settings, host));
        return out[0];
    }

    @SuppressWarnings("unchecked")
    static Map<String, JCheckBox> field(RegistrationDialog d, String name) throws Exception {
        Field f = RegistrationDialog.class.getDeclaredField(name); f.setAccessible(true); return (Map<String, JCheckBox>) f.get(d);
    }

    public static void main(String[] args) {
        // A failed check must end the JVM: the dialog's Swing thread would otherwise keep it alive.
        try { run(); System.out.println("Dialog wiring checks passed."); System.exit(0); }
        catch (Throwable failure) { failure.printStackTrace(); System.exit(1); }
    }

    static void run() throws Exception {
        // 1. Catalog present: the toggle is shown, collapsed; simple boxes mirror into the list.
        RegistrationSettings s = RegistrationSettings.defaults(ConnectorSmokeTest.SPEC_DEFAULTS);
        s.model = "openai-oauth/gpt-6-sol"; s.fresh = false;
        RegistrationDialog d = dialog(status(), s);
        Map<String, JCheckBox> mirrors = field(d, "mirrors"), boxes = field(d, "toolBoxes");
        require(d.advanced.isVisible() && !d.advancedPanel.isVisible(), "Advanced toggle shown, panel collapsed");
        require(mirrors.keySet().equals(new HashSet<>(Arrays.asList("mark_damage", "elastix_affine", "interactive_transform:flip"))), "Mirrors: " + mirrors.keySet());
        require(boxes.keySet().equals(new HashSet<>(Arrays.asList("zoom", "fireants_syn", "look:section", "look:atlas", "interactive_transform"))), "Catalog boxes: " + boxes.keySet());
        require(d.read().spec().getAsJsonArray("disabled").size() == 0, "Nothing off by default");
        // 2. Simple box drives its mirror; the mirror drives the simple box.
        d.agentDamage.doClick();
        require(!mirrors.get("mark_damage").isSelected(), "Simple box off -> mirror off");
        require(d.read().spec().getAsJsonArray("disabled").toString().equals("[\"mark_damage\"]"), "Damage off is mark_damage");
        mirrors.get("mark_damage").doClick();
        require(d.agentDamage.isSelected() && mirrors.get("mark_damage").isSelected(), "Mirror click turns the simple box back on");
        mirrors.get("elastix_affine").doClick();
        require(!d.affine.isSelected() && d.read().spec().getAsJsonArray("disabled").toString().equals("[\"elastix_affine\"]"), "Mirror off -> affine off");
        d.affine.doClick();
        // 3. A catalog checkbox lands in disabled; the flip mirror too.
        boxes.get("zoom").doClick();
        boxes.get("fireants_syn").doClick();
        d.flip.doClick();
        JsonArray names = d.read().spec().getAsJsonArray("disabled");
        require(names.toString().equals("[\"interactive_transform:flip\",\"zoom\",\"fireants_syn\"]"), "Disabled: " + names);
        require(!d.read().spec().has("agent_damage") && !d.read().spec().getAsJsonObject("transform").has("flip"), "Old keys absent from the spec");
        // 4. Save and reopen: the Advanced choices come back.
        java.util.prefs.Preferences node = java.util.prefs.Preferences.userRoot().node("dialogcheck-" + System.nanoTime());
        RegistrationSettings saved = d.read();
        saved.save(node);
        RegistrationSettings back = RegistrationSettings.load(node, ConnectorSmokeTest.SPEC_DEFAULTS);
        RegistrationDialog again = dialog(status(), back);
        Map<String, JCheckBox> againBoxes = field(again, "toolBoxes");
        require(!againBoxes.get("zoom").isSelected() && !againBoxes.get("fireants_syn").isSelected() && againBoxes.get("look:section").isSelected(),
                "Advanced checkboxes restored from settings");
        require(!again.flip.isSelected() && field(again, "mirrors").get("interactive_transform:flip").isSelected() == false,
                "Flip restored, and its mirror follows");
        require(again.read().spec().getAsJsonArray("disabled").toString().equals("[\"interactive_transform:flip\",\"zoom\",\"fireants_syn\"]"), "Restored disabled list");
        node.removeNode();
    }
}
