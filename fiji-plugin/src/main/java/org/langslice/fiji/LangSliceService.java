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
    static final String SETUP = "LangSlice>LangSlice setup…";
    static final String AGENT = "LangSlice>LangSlice agent…";
    private static boolean registered;
    @Parameter private PluginService plugins;
    @Override public void initialize() {
        // ABBA uses CommandService.run(String), which looks up class identifiers,
        // not the human-readable @Plugin name. Supply explicit callable aliases.
        installAlias(SETUP, SetupCommand.class);
        installAlias(AGENT, AgentCommand.class);
        synchronized (MultiSlicePositioner.class) {
            if (registered) return;
            MultiSlicePositioner.registerRegistrationPluginUI("LangSlice connector", SETUP);
            MultiSlicePositioner.registerRegistrationPluginUI("LangSlice connector", AGENT);
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
