package com.example.yo7a.healthwatcher;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.example.yo7a.healthwatcher.ml.CapabilityItem;
import com.example.yo7a.healthwatcher.ml.MLModelManager;
import com.example.yo7a.healthwatcher.ml.PipelineSelfTest;
import com.example.yo7a.healthwatcher.ml.RuntimeSelfTest;
import com.example.yo7a.healthwatcher.ml.VitalTarget;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * "System check": runs real checks on this device and reports what is actually available - camera, flash,
 * permission, PPG pipeline, TensorFlow Lite runtime, each ML model, local database and validation status.
 * Nothing here is hard-coded as "OK".
 */
@SuppressWarnings("deprecation")
public class CapabilityActivity extends AppCompatActivity {
    private String user = "";
    private ExecutorService io;
    private volatile boolean destroyed = false;
    private LinearLayout container;
    private TextView summary;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_capability);
        Bundle b = getIntent().getExtras();
        if (b != null && b.getString(MeasurementActivity.EXTRA_USER) != null) {
            user = b.getString(MeasurementActivity.EXTRA_USER);
        }
        container = findViewById(R.id.capContainer);
        summary = findViewById(R.id.capSummary);
        findViewById(R.id.capRerun).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runChecks();
            }
        });
        io = Executors.newSingleThreadExecutor();
    }

    @Override
    protected void onResume() {
        super.onResume();
        runChecks();     // also refreshes the permission row after returning from system settings
    }

    private void runChecks() {
        summary.setText("Running checks\u2026");
        container.removeAllViews();
        io.execute(new Runnable() {
            @Override
            public void run() {
                final List<CapabilityItem> items = collect();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (!destroyed) {
                            render(items);
                        }
                    }
                });
            }
        });
    }

    private List<CapabilityItem> collect() {
        List<CapabilityItem> out = new ArrayList<>();
        PackageManager pm = getPackageManager();

        // Camera hardware
        int back = 0;
        try {
            Camera.CameraInfo ci = new Camera.CameraInfo();
            for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
                Camera.getCameraInfo(i, ci);
                if (ci.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                    back++;
                }
            }
        } catch (RuntimeException ignored) {
            // treated as no camera
        }
        out.add(back > 0 && pm.hasSystemFeature(PackageManager.FEATURE_CAMERA)
                ? new CapabilityItem("Rear camera", CapabilityItem.Status.PASS, "A rear camera is present.")
                : new CapabilityItem("Rear camera", CapabilityItem.Status.FAIL, "No rear camera found; measurements are impossible on this device."));
        out.add(pm.hasSystemFeature(PackageManager.FEATURE_CAMERA_FLASH)
                ? new CapabilityItem("Camera flash (torch)", CapabilityItem.Status.PASS, "Flash hardware is present.")
                : new CapabilityItem("Camera flash (torch)", CapabilityItem.Status.FAIL, "No flash; the fingertip cannot be lit, so measurements are not possible."));
        boolean granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        out.add(granted
                ? new CapabilityItem("Camera permission", CapabilityItem.Status.PASS, "Granted.")
                : new CapabilityItem("Camera permission", CapabilityItem.Status.WARN, "Not granted yet. You will be asked when you start a measurement."));

        // PPG pipeline and TensorFlow Lite runtime
        out.add(PipelineSelfTest.run());
        MLModelManager.RunnerFactory factory = new MLModelManager.RunnerFactory() {
            @Override
            public MLModelManager.ModelRunner create(java.nio.ByteBuffer model) {
                return new com.example.yo7a.healthwatcher.ml.TfliteModelRunner(model);
            }
        };
        try (InputStream in = getAssets().open(RuntimeSelfTest.ASSET)) {
            out.add(RuntimeSelfTest.run(in, factory));
        } catch (java.io.IOException e) {
            out.add(new CapabilityItem("On-device inference (TensorFlow Lite runtime)", CapabilityItem.Status.FAIL,
                    "Self-test model is missing from the app package."));
        }

        // Installed models
        MLModelManager m = ModelStatus.newManager(this);
        boolean anyValidated = false;
        int ready = 0;
        try {
            m.loadAll();
            for (VitalTarget t : VitalTarget.values()) {
                MLModelManager.ModelInfo i = m.info(t);
                String title = "ML model: " + t.displayName;
                switch (i.state) {
                    case READY:
                        ready++;
                        if (i.manifest.isValidated()) {
                            anyValidated = true;
                            out.add(new CapabilityItem(title, CapabilityItem.Status.PASS,
                                    "Validated model " + i.manifest.modelVersion + " loaded; used as the main result."));
                        } else {
                            out.add(new CapabilityItem(title, CapabilityItem.Status.WARN,
                                    "Model " + i.manifest.modelVersion + " loaded but NOT validated; shown only as an experimental comparison."));
                        }
                        break;
                    case NOT_INSTALLED:
                        out.add(new CapabilityItem(title, CapabilityItem.Status.INFO,
                                "No trained model installed. The original signal-processing algorithm is used."));
                        break;
                    default:
                        out.add(new CapabilityItem(title, CapabilityItem.Status.FAIL,
                                "Model present but unusable: " + i.detail));
                        break;
                }
            }
        } finally {
            m.close();
        }
        out.add(anyValidated
                ? new CapabilityItem("Validation status", CapabilityItem.Status.PASS, "At least one validated model is in use.")
                : new CapabilityItem("Validation status", CapabilityItem.Status.WARN,
                        ready > 0 ? "No validated model: every displayed value is signal-processing or experimental."
                                : "No ML model installed: every displayed value comes from the original signal-processing algorithms. "
                                + "Blood pressure and SpO₂ are experimental formulas."));
        com.example.yo7a.healthwatcher.ml.ScreeningClassifier screening = ModelStatus.newScreening(this);
        try {
            out.add(screening.selfTest());
        } finally {
            screening.close();
        }

        // Database
        String probe = "capability_probe.db";
        try {
            UserDB db = new UserDB(this, probe);
            MeasurementRecord r = new MeasurementRecord();
            r.username = "probe";
            r.timestampMillis = System.currentTimeMillis();
            r.heartRate = 60;
            boolean ok = db.addMeasurement(r) > 0 && db.getMeasurements("probe", 5).size() == 1
                    && db.deleteMeasurements("probe") == 1;
            db.close();
            deleteDatabase(probe);
            int stored = new UserDB(this).getMeasurements(user, 1000).size();
            out.add(ok ? new CapabilityItem("Local database", CapabilityItem.Status.PASS,
                    "Read/write test passed. " + stored + " measurement(s) stored for this account.")
                    : new CapabilityItem("Local database", CapabilityItem.Status.FAIL, "Read/write test failed."));
        } catch (RuntimeException e) {
            out.add(new CapabilityItem("Local database", CapabilityItem.Status.FAIL, "Database error: " + e.getMessage()));
        }
        return out;
    }

    private void render(List<CapabilityItem> items) {
        LayoutInflater inf = LayoutInflater.from(this);
        int fail = 0, warn = 0, pass = 0;
        container.removeAllViews();
        for (CapabilityItem it : items) {
            View row = inf.inflate(R.layout.item_capability, container, false);
            ((TextView) row.findViewById(R.id.capTitle)).setText(it.title);
            ((TextView) row.findViewById(R.id.capDetail)).setText(it.detail);
            TextView badge = row.findViewById(R.id.capBadge);
            switch (it.status) {
                case PASS:
                    badge.setText("PASS");
                    badge.setBackgroundResource(R.drawable.bg_ok);
                    badge.setTextColor(getResources().getColor(R.color.hw_ok_text));
                    pass++;
                    break;
                case WARN:
                    badge.setText("ATTENTION");
                    badge.setBackgroundResource(R.drawable.bg_warn);
                    badge.setTextColor(getResources().getColor(R.color.hw_warn_text));
                    warn++;
                    break;
                case FAIL:
                    badge.setText("FAIL");
                    badge.setBackgroundResource(R.drawable.bg_error);
                    badge.setTextColor(getResources().getColor(R.color.hw_error_text));
                    fail++;
                    break;
                default:
                    badge.setText("INFO");
                    badge.setBackgroundResource(R.drawable.bg_badge);
                    badge.setTextColor(getResources().getColor(R.color.hw_text));
                    break;
            }
            row.setContentDescription(it.title + ": " + badge.getText() + ". " + it.detail);
            container.addView(row);
        }
        summary.setText(pass + " passed · " + warn + " need attention \u00b7 " + fail + " failed");
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        io.shutdownNow();
        super.onDestroy();
    }

    @Override
    public void onBackPressed() {
        Intent i = new Intent(this, Primary.class);
        i.putExtra(MeasurementActivity.EXTRA_USER, user);
        i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
        startActivity(i);
        finish();
    }
}
