package org.langslice.fiji;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;
@Plugin(type = Command.class, name = LangSliceService.SETUP, menuPath = "Plugins>LangSlice>LangSlice setup…")
public final class SetupCommand implements Command {
    @Override public void run() { SetupDialog.open(); }
}
