package org.langslice.fiji;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;
@Plugin(type = Command.class, name = LangSliceService.ENVIRONMENT, menuPath = "Plugins>LangSlice>Choose Python environment…")
public final class SetupCommand implements Command {
    @Override public void run() { EnvironmentDialog.open(); }
}
