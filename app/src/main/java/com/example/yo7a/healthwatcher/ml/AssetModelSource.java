package com.example.yo7a.healthwatcher.ml;

import android.content.res.AssetManager;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

/** {@link MLModelManager.ModelSource} backed by {@code assets/models/}. */
public final class AssetModelSource implements MLModelManager.ModelSource {
    public static final String DIR = "models";
    private final AssetManager assets;
    private List<String> names;

    public AssetModelSource(AssetManager assets) {
        this.assets = assets;
    }

    @Override
    public boolean exists(String fileName) {
        try {
            if (names == null) {
                String[] list = assets.list(DIR);
                names = list == null ? java.util.Collections.<String>emptyList() : Arrays.asList(list);
            }
            return names.contains(fileName);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public InputStream open(String fileName) throws IOException {
        return assets.open(DIR + "/" + fileName);
    }
}
