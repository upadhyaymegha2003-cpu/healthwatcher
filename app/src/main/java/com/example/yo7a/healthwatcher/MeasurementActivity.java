package com.example.yo7a.healthwatcher;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.yo7a.healthwatcher.ml.MLModelManager;
import com.example.yo7a.healthwatcher.ml.PpgRecorder;
import com.example.yo7a.healthwatcher.ml.PpgSignal;
import com.example.yo7a.healthwatcher.ml.SignalQualityAnalyzer;
import com.example.yo7a.healthwatcher.ml.UserProfile;
import com.example.yo7a.healthwatcher.ml.VitalSignsPredictor;
import com.example.yo7a.healthwatcher.ml.VitalSignsReport;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Shared camera + PPG measurement screen. Replaces the five copy-pasted measurement activities:
 * camera/flash handling, runtime permission, finger-aware 30 s recording, off-UI-thread frame
 * decoding and analysis, quality-gated results, retry, and correct release of camera and models.
 * Subclasses only choose the {@link MeasurementMode}.
 */
@SuppressWarnings("deprecation")   // android.hardware.Camera (API 1) is what the existing app uses
public abstract class MeasurementActivity extends AppCompatActivity
        implements SurfaceHolder.Callback, Camera.PreviewCallback {

    public static final String EXTRA_USER = "Usr";
    public static final String EXTRA_REPORT = "report";
    private static final int REQ_CAMERA = 42;
    private static final double RECORD_SECONDS = 30.0;
    private static final int FINGER_FRAMES_TO_START = 10;
    private static final double FINGER_LOST_SECONDS = 1.0;
    private static final double FINGER_RED_MIN = 200.0;

    private enum State {WAITING_FINGER, RECORDING, ANALYZING, ERROR}

    protected abstract MeasurementMode mode();

    private final Object lock = new Object();
    private State state = State.WAITING_FINGER;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final PpgRecorder recorder = new PpgRecorder();
    private final SignalQualityAnalyzer analyzer = new SignalQualityAnalyzer();
    private final AtomicBoolean frameBusy = new AtomicBoolean(false);
    private ExecutorService frameExecutor;
    private ExecutorService workExecutor;
    private volatile boolean destroyed = false;

    private MLModelManager models;
    private UserProfile profile;
    private String user = "";

    private Camera camera;
    private SurfaceHolder holder;
    private boolean surfaceReady = false;
    private boolean previewing = false;
    private int previewW, previewH;

    private int goodStreak = 0;
    private double recordStartT = 0;
    private double lastLowT = Double.NaN;
    private double lastUiUpdateT = 0;
    private double lastLiveT = 0;

    private ProgressBar progress;
    private TextView statusText, qualityText, instructionText, modelText, errorText;
    private View errorPanel;
    private Button retryButton, cancelButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_measurement);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        String u = getIntent().getStringExtra(EXTRA_USER);
        user = u == null ? "" : u;

        SurfaceView preview = findViewById(R.id.preview);
        holder = preview.getHolder();
        holder.addCallback(this);
        progress = findViewById(R.id.measureProgress);
        statusText = findViewById(R.id.statusText);
        qualityText = findViewById(R.id.qualityText);
        instructionText = findViewById(R.id.instructionText);
        modelText = findViewById(R.id.modelText);
        errorPanel = findViewById(R.id.errorPanel);
        errorText = findViewById(R.id.errorText);
        retryButton = findViewById(R.id.retryButton);
        cancelButton = findViewById(R.id.cancelButton);
        ((TextView) findViewById(R.id.measureTitle)).setText(titleFor(mode()));

        retryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                retry();
            }
        });
        cancelButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                goHome();
            }
        });

        profile = loadProfile();
        frameExecutor = Executors.newSingleThreadExecutor();
        workExecutor = Executors.newSingleThreadExecutor();
        models = ModelStatus.newManager(this);
        workExecutor.execute(new Runnable() {
            @Override
            public void run() {
                models.loadAll();
            }
        });
        progress.setMax(100);
        showWaiting();
    }

    private static String titleFor(MeasurementMode m) {
        switch (m) {
            case HEART_RATE:
                return "Heart rate";
            case BLOOD_PRESSURE:
                return "Blood pressure";
            case RESPIRATION:
                return "Respiratory rate";
            case SPO2:
                return "Oxygen saturation";
            default:
                return "Full measurement";
        }
    }

    private UserProfile loadProfile() {
        try {
            UserDB db = new UserDB(this);
            return new UserProfile(Integer.parseInt(db.getage(user)), Integer.parseInt(db.getheight(user)),
                    Integer.parseInt(db.getweight(user)), Integer.parseInt(db.getgender(user)));
        } catch (RuntimeException e) {   // unknown user / non-numeric data: profile-dependent estimates are skipped
            return null;
        }
    }

    // ------------------------------------------------------------------ lifecycle

    @Override
    protected void onResume() {
        super.onResume();
        if (state == State.ERROR || state == State.ANALYZING) {
            return;
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQ_CAMERA) {
            return;
        }
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            showError("Camera permission is required to measure your pulse. "
                    + "Allow it when asked, or enable it in the system settings for this app.", true);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        stopCamera();
        synchronized (lock) {
            if (state == State.RECORDING) {      // an interrupted recording is never analysed
                recorder.clear();
                state = State.WAITING_FINGER;
            }
        }
        goodStreak = 0;
        if (state == State.WAITING_FINGER) {
            showWaiting();
        }
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        ui.removeCallbacksAndMessages(null);
        stopCamera();
        if (frameExecutor != null) {
            frameExecutor.shutdownNow();
        }
        if (workExecutor != null) {
            workExecutor.shutdownNow();
        }
        if (models != null) {
            models.close();
        }
        super.onDestroy();
    }

    @Override
    public void onBackPressed() {
        goHome();
    }

    private void goHome() {
        Intent i = new Intent(this, Primary.class);
        i.putExtra(EXTRA_USER, user);
        i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
        startActivity(i);
        finish();
    }

    // ------------------------------------------------------------------ camera

    private void startCamera() {
        if (camera != null) {
            maybeStartPreview();
            return;
        }
        try {
            camera = Camera.open();
        } catch (RuntimeException e) {
            camera = null;
            showError("The camera could not be opened. Close other apps that use the camera and try again.", false);
            return;
        }
        if (camera == null) {
            showError("This device has no usable back camera.", false);
            return;
        }
        try {
            Camera.Parameters p = camera.getParameters();
            List<String> flash = p.getSupportedFlashModes();
            if (flash == null || !flash.contains(Camera.Parameters.FLASH_MODE_TORCH)) {
                stopCamera();
                showError("This device has no camera flash (torch), which the measurement needs to light your fingertip.", false);
                return;
            }
            p.setFlashMode(Camera.Parameters.FLASH_MODE_TORCH);
            Camera.Size best = null;
            for (Camera.Size s : p.getSupportedPreviewSizes()) {
                if (best == null || s.width * s.height < best.width * best.height) {
                    best = s;
                }
            }
            if (best != null) {
                p.setPreviewSize(best.width, best.height);
            }
            camera.setParameters(p);
            Camera.Size size = camera.getParameters().getPreviewSize();
            previewW = size.width;
            previewH = size.height;
            camera.setDisplayOrientation(90);
        } catch (RuntimeException e) {
            stopCamera();
            showError("The camera or flash could not be configured on this device.", false);
            return;
        }
        maybeStartPreview();
    }

    private void maybeStartPreview() {
        if (camera == null || !surfaceReady || previewing) {
            return;
        }
        try {
            camera.setPreviewDisplay(holder);
            camera.setPreviewCallback(this);
            camera.startPreview();
            previewing = true;
        } catch (Exception e) {
            stopCamera();
            showError("The camera preview could not be started.", false);
        }
    }

    private void stopCamera() {
        Camera c = camera;
        camera = null;
        previewing = false;
        if (c == null) {
            return;
        }
        try {
            c.setPreviewCallback(null);
            c.stopPreview();
        } catch (RuntimeException ignored) {
            // camera already gone
        }
        try {
            Camera.Parameters p = c.getParameters();
            p.setFlashMode(Camera.Parameters.FLASH_MODE_OFF);
            c.setParameters(p);
        } catch (RuntimeException ignored) {
            // best effort: release() also turns the torch off
        }
        c.release();
    }

    @Override
    public void surfaceCreated(SurfaceHolder h) {
        surfaceReady = true;
        maybeStartPreview();
    }

    @Override
    public void surfaceChanged(SurfaceHolder h, int format, int width, int height) {
        // fixed-size preview, nothing to do
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder h) {
        surfaceReady = false;
        previewing = false;
    }

    // ------------------------------------------------------------------ frames

    @Override
    public void onPreviewFrame(byte[] data, Camera cam) {
        if (data == null || previewW == 0) {
            return;
        }
        synchronized (lock) {
            if (state != State.WAITING_FINGER && state != State.RECORDING) {
                return;
            }
        }
        if (!frameBusy.compareAndSet(false, true)) {
            return;       // drop the frame instead of queueing work behind a slow decode
        }
        final double t = SystemClock.elapsedRealtimeNanos() / 1e9;
        final byte[] copy = data.clone();   // the camera reuses its buffer
        try {
            frameExecutor.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        // (height, width) order is intentional: it matches the original app's decoding.
                        double[] rgb = ImageProcessing.decodeYUV420SPtoRedGreenBlueMeans(copy, previewH, previewW);
                        onFrame(t, rgb);
                    } finally {
                        frameBusy.set(false);
                    }
                }
            });
        } catch (RejectedExecutionException e) {
            frameBusy.set(false);
        }
    }

    private void onFrame(double t, double[] rgb) {
        boolean finished = false;
        synchronized (lock) {
            boolean finger = rgb[0] >= FINGER_RED_MIN;
            if (state == State.WAITING_FINGER) {
                goodStreak = finger ? goodStreak + 1 : 0;
                if (goodStreak >= FINGER_FRAMES_TO_START) {
                    recorder.clear();
                    recordStartT = t;
                    lastLowT = Double.NaN;
                    state = State.RECORDING;
                    postRecordingStarted();
                }
            }
            if (state == State.RECORDING) {
                recorder.add(t, rgb[0], rgb[1], rgb[2]);
                if (finger) {
                    lastLowT = Double.NaN;
                } else {
                    if (Double.isNaN(lastLowT)) {
                        lastLowT = t;
                    }
                    if (t - lastLowT > FINGER_LOST_SECONDS) {
                        recorder.clear();
                        goodStreak = 0;
                        state = State.WAITING_FINGER;
                        postFingerLost();
                        return;
                    }
                }
                double elapsed = t - recordStartT;
                if (t - lastUiUpdateT > 0.25) {
                    lastUiUpdateT = t;
                    postProgress(elapsed);
                }
                if (t - lastLiveT > 1.0) {
                    lastLiveT = t;
                    double[] m = recorder.recentMeans(1.5);
                    postLive(analyzer.assessLive(m[0], m[1]));
                }
                if (elapsed >= RECORD_SECONDS) {
                    state = State.ANALYZING;
                    finished = true;
                }
            }
        }
        if (finished) {
            startAnalysis();
        }
    }

    // ------------------------------------------------------------------ analysis

    private void startAnalysis() {
        ui.post(new Runnable() {
            @Override
            public void run() {
                if (destroyed) {
                    return;
                }
                progress.setProgress(100);
                statusText.setText("Analysing your recording\u2026");
                qualityText.setText("");
                instructionText.setText("You can lift your finger now.");
            }
        });
        final PpgSignal signal = recorder.snapshot();
        final UserProfile prof = profile;
        try {
            workExecutor.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        models.loadAll();
                        VitalSignsReport report = new VitalSignsPredictor(models, analyzer)
                                .predict(signal, prof, mode().targets());
                        if (report.isSuccessful()) {
                            runScreening(report, prof);
                            try {
                                new UserDB(MeasurementActivity.this)
                                        .addMeasurement(MeasurementRecord.fromReport(user, report));
                            } catch (RuntimeException e) {
                                // history is a convenience; never lose the result because storage failed
                            }
                        }
                        postReport(report);
                    } catch (final RuntimeException e) {
                        ui.post(new Runnable() {
                            @Override
                            public void run() {
                                if (!destroyed) {
                                    showError("The analysis failed unexpectedly (" + e.getClass().getSimpleName()
                                            + "). Please retry.", false);
                                }
                            }
                        });
                    }
                }
            });
        } catch (RejectedExecutionException ignored) {
            // activity is being destroyed
        }
    }

    /** Adds the ML health-screening verdict (good / needs review / high priority) to a successful report. */
    private void runScreening(VitalSignsReport report, UserProfile prof) {
        com.example.yo7a.healthwatcher.ml.ScreeningClassifier sc = null;
        try {
            sc = ModelStatus.newScreening(this);
            report.screening = sc.classify(report, prof);
        } catch (RuntimeException | Error e) {
            // screening is an add-on: a failure must never cost the person their vital-sign result
            report.screening = null;
        } finally {
            if (sc != null) {
                sc.close();
            }
        }
    }

    private void postReport(final VitalSignsReport report) {
        ui.post(new Runnable() {
            @Override
            public void run() {
                if (destroyed) {
                    return;
                }
                if (report.isSuccessful()) {
                    Intent i = new Intent(MeasurementActivity.this, mode().resultActivity);
                    i.putExtra(EXTRA_USER, user);
                    i.putExtra(EXTRA_REPORT, report);
                    startActivity(i);
                    finish();
                } else {
                    showError(report.failureMessage(), false);
                }
            }
        });
    }

    // ------------------------------------------------------------------ UI state (main thread only)

    private void retry() {
        synchronized (lock) {
            recorder.clear();
            goodStreak = 0;
            state = State.WAITING_FINGER;
        }
        errorPanel.setVisibility(View.GONE);
        showWaiting();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        }
    }

    private void showWaiting() {
        progress.setProgress(0);
        statusText.setText("Place your fingertip on the camera");
        qualityText.setText("Signal: no finger detected");
        instructionText.setText("Cover the rear camera lens AND the flash completely with your fingertip. "
                + "Press gently, keep your hand still and rest it on a table.");
        modelText.setText("");
        workExecutorDescribe();
    }

    private void workExecutorDescribe() {
        if (workExecutor == null || workExecutor.isShutdown()) {
            return;
        }
        try {
            workExecutor.execute(new Runnable() {
                @Override
                public void run() {
                    models.loadAll();
                    boolean any = false;
                    boolean validated = false;
                    for (com.example.yo7a.healthwatcher.ml.VitalTarget t : mode().targets()) {
                        MLModelManager.ModelInfo i = models.info(t);
                        if (i.isReady()) {
                            any = true;
                            validated |= i.manifest.isValidated();
                        }
                    }
                    final String msg = !any ? "Method: signal processing (no trained model installed)"
                            : validated ? "Method: validated on-device ML model"
                            : "Method: signal processing; installed ML model is experimental and shown for comparison only";
                    ui.post(new Runnable() {
                        @Override
                        public void run() {
                            if (!destroyed) {
                                modelText.setText(msg);
                            }
                        }
                    });
                }
            });
        } catch (RejectedExecutionException ignored) {
            // shutting down
        }
    }

    private void showError(String message, boolean permissionProblem) {
        synchronized (lock) {
            state = State.ERROR;
        }
        errorText.setText(message);
        errorPanel.setVisibility(View.VISIBLE);
        statusText.setText(permissionProblem ? "Camera permission needed" : "Measurement not completed");
        progress.setProgress(0);
        qualityText.setText("");
        instructionText.setText("");
    }

    private void postRecordingStarted() {
        ui.post(new Runnable() {
            @Override
            public void run() {
                if (!destroyed) {
                    statusText.setText("Recording \u2013 keep still");
                    progress.setProgress(0);
                }
            }
        });
    }

    private void postFingerLost() {
        ui.post(new Runnable() {
            @Override
            public void run() {
                if (!destroyed) {
                    progress.setProgress(0);
                    statusText.setText("Finger moved \u2013 restarting");
                    qualityText.setText("Signal: no finger detected");
                    instructionText.setText("Cover the camera and flash completely, then hold still.");
                }
            }
        });
    }

    private void postProgress(final double elapsed) {
        ui.post(new Runnable() {
            @Override
            public void run() {
                if (destroyed) {
                    return;
                }
                int pct = (int) Math.min(100, Math.round(elapsed / RECORD_SECONDS * 100));
                progress.setProgress(pct);
                int left = (int) Math.max(0, Math.ceil(RECORD_SECONDS - elapsed));
                statusText.setText("Recording \u2013 " + left + " s left");
            }
        });
    }

    private void postLive(final SignalQualityAnalyzer.LiveState s) {
        ui.post(new Runnable() {
            @Override
            public void run() {
                if (destroyed) {
                    return;
                }
                switch (s) {
                    case OK:
                        qualityText.setText("Signal: finger detected");
                        instructionText.setText("Good. Keep your finger still until the bar is full.");
                        break;
                    case SATURATED:
                        qualityText.setText("Signal: image too bright");
                        instructionText.setText("Press more lightly and avoid direct bright light.");
                        break;
                    default:
                        qualityText.setText("Signal: no finger detected");
                        instructionText.setText("Cover the camera lens and flash completely.");
                        break;
                }
            }
        });
    }
}
