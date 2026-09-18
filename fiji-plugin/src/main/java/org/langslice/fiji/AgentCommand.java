package org.langslice.fiji;
import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import javax.swing.SwingUtilities;
import org.scijava.command.Command;
import org.scijava.plugin.*;
@Plugin(type = Command.class, name = LangSliceService.AGENT)
public final class AgentCommand implements Command {
    @Parameter private MultiSlicePositioner mp;
    @Override public void run() { SwingUtilities.invokeLater(() -> SetupDialog.ensureConfigured(mp, () -> AgentRunner.show(mp, EnvironmentDiscovery.saved()))); }
}
