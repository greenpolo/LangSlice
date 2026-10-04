package org.langslice.fiji;

import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import org.scijava.plugin.Plugin;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.PluginService;
import org.scijava.command.Command;
import org.scijava.command.CommandInfo;
import org.scijava.service.AbstractService;
import org.scijava.service.Service;

/** Install before ABBA builds its registration menu; no Python dependency at startup. */
@Plugin(type = Service.class)
public final class LangSliceService extends AbstractService {
    /** ABBA groups external entries under this key; the key itself is never shown. */
    static final String MENU = "LangSlice";
    /**
     * ABBA shows each registered command string as "Register>" + string: one plain entry in the Register menu,
     * after a separator below ABBA's own entries (a plugin cannot choose the position).
     */
    static final String REGISTRATION = "LangSlice Registration…";
    /** Setup lives in Fiji's Plugins>LangSlice menu and behind the dialog's Setup… button. */
    static final String SETUP = "LangSlice setup";
    private static boolean registered;
    @Parameter private PluginService plugins;
    @Override public void initialize() {
        // ABBA uses CommandService.run(String), which looks up class identifiers,
        // not the human-readable @Plugin name. Supply an explicit callable alias.
        installAlias(REGISTRATION, RegistrationCommand.class);
        synchronized (MultiSlicePositioner.class) {
            if (registered) return;
            MultiSlicePositioner.registerRegistrationPluginUI(MENU, REGISTRATION);
            registered = true;
        }
    }
    private void installAlias(final String alias, Class<? extends Command> command) {
        if (plugins.getPluginsOfClass(alias).isEmpty()) {
            plugins.addPlugin(new CommandInfo(command) {
                @Override public String getClassName() { return alias; }
            });
        }
    }
}
