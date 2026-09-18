package org.langslice.fiji;
import ch.epfl.biop.atlas.aligner.MultiSlicePositioner;
import org.scijava.command.Command;
import org.scijava.plugin.*;
@Plugin(type = Command.class, name = LangSliceService.SETUP, menuPath = "Plugins>LangSlice>LangSlice setup…")
public final class SetupCommand implements Command {
    @Parameter(required = false) private MultiSlicePositioner mp;
    @Override public void run() { SetupDialog.show(mp); }
}
