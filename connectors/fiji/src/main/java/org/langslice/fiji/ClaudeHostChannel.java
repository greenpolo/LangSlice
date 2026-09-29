package org.langslice.fiji;

import com.google.gson.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.function.Consumer;

/** One authenticated, loopback-only connection. No scripts or host commands are accepted. */
final class ClaudeHostChannel implements AutoCloseable {
    private final ServerSocket listener;
    private final String token;
    private volatile Socket client;
    private volatile boolean closed;

    ClaudeHostChannel() throws IOException {
        listener = new ServerSocket(0, 8, InetAddress.getByName("127.0.0.1"));
        byte[] secret = new byte[32]; new SecureRandom().nextBytes(secret);
        token = Base64.getUrlEncoder().withoutPadding().encodeToString(secret);
    }

    JsonObject settings() {
        JsonObject settings = new JsonObject();
        settings.addProperty("address", "127.0.0.1");
        settings.addProperty("port", listener.getLocalPort()); settings.addProperty("token", token);
        return settings;
    }

    /** Wrong tokens do not consume the listener; the first valid token consumes it forever. */
    JsonObject receive(Consumer<JsonObject> events) throws IOException {
        while (!closed) {
            Socket candidate = listener.accept(); client = candidate;
            try {
                candidate.setSoTimeout(3000);
                BufferedReader reader = new BufferedReader(new InputStreamReader(candidate.getInputStream(), StandardCharsets.UTF_8));
                String hello = readLine(reader, 4096);
                JsonObject auth;
                try { auth = JsonParser.parseString(hello == null ? "{}" : hello).getAsJsonObject(); }
                catch (RuntimeException malformed) { continue; }
                if (!auth.has("token") || !auth.get("token").isJsonPrimitive()
                        || !MessageDigest.isEqual(token.getBytes(StandardCharsets.UTF_8), auth.get("token").getAsString().getBytes(StandardCharsets.UTF_8))) continue;
                listener.close(); candidate.setSoTimeout(0);
                String line;
                while (!closed && (line = readLine(reader, 16 * 1024 * 1024)) != null) {
                    JsonObject envelope = JsonParser.parseString(line).getAsJsonObject();
                    String type = envelope.has("type") ? envelope.get("type").getAsString() : "";
                    if ("event".equals(type)) events.accept(envelope.getAsJsonObject("event"));
                    else if ("result".equals(type)) return envelope.getAsJsonObject("result");
                }
                throw new EOFException("Claude disconnected. Saved results remain in the LangSlice job directory.");
            } catch (SocketTimeoutException unauthenticated) {
                // A silent connection cannot occupy the listener indefinitely.
            } finally { candidate.close(); client = null; }
        }
        throw new IOException("Claude listener closed");
    }

    private static String readLine(Reader reader, int limit) throws IOException {
        StringBuilder line = new StringBuilder(); int ch;
        while ((ch = reader.read()) != -1) {
            if (ch == '\n') return line.toString();
            if (line.length() >= limit) throw new IOException("Host channel message exceeds its size limit");
            line.append((char) ch);
        }
        return line.length() == 0 ? null : line.toString();
    }

    @Override public void close() {
        closed = true;
        try { listener.close(); } catch (IOException ignored) { }
        Socket socket = client;
        if (socket != null) try { socket.close(); } catch (IOException ignored) { }
    }
}
