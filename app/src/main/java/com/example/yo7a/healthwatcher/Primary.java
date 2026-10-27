package com.example.yo7a.healthwatcher;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.example.yo7a.healthwatcher.ml.VitalTarget;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Home screen: four vital-sign cards (latest stored real values), start button, history, about. */
public class Primary extends AppCompatActivity {
    private String user = "";
    private ExecutorService io;
    private volatile boolean destroyed = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_primary);
        Bundle extras = getIntent().getExtras();
        if (extras != null && extras.getString(MeasurementActivity.EXTRA_USER) != null) {
            user = extras.getString(MeasurementActivity.EXTRA_USER);
        }
        ((TextView) findViewById(R.id.welcomeText)).setText("Hello, " + user);
        io = Executors.newSingleThreadExecutor();

        bindCard(R.id.cardHeart, MeasurementMode.HEART_RATE);
        bindCard(R.id.cardBp, MeasurementMode.BLOOD_PRESSURE);
        bindCard(R.id.cardResp, MeasurementMode.RESPIRATION);
        bindCard(R.id.cardSpo2, MeasurementMode.SPO2);

        findViewById(R.id.startAllButton).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                open(StartVitalSigns.class, MeasurementMode.ALL.page);
            }
        });
        findViewById(R.id.historyButton).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                open(HistoryActivity.class, 0);
            }
        });
        findViewById(R.id.capabilityButton).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                open(CapabilityActivity.class, 0);
            }
        });
        findViewById(R.id.aboutButton).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                open(AboutApp.class, 0);
            }
        });
    }

    private void bindCard(int id, final MeasurementMode mode) {
        findViewById(id).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                open(StartVitalSigns.class, mode.page);
            }
        });
    }

    private void open(Class<?> cls, int page) {
        Intent i = new Intent(this, cls);
        i.putExtra(MeasurementActivity.EXTRA_USER, user);
        i.putExtra("Page", page);
        startActivity(i);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshLatest();
    }

    /** Shows the most recent REAL stored value per vital; "--" when nothing has been measured yet. */
    private void refreshLatest() {
        final String u = user;
        io.execute(new Runnable() {
            @Override
            public void run() {
                String hr = "--", bp = "--", rr = "--", sp = "--";
                try {
                    List<MeasurementRecord> list = new UserDB(Primary.this).getMeasurements(u, 100);
                    boolean h = false, b = false, r = false, s = false;
                    for (MeasurementRecord m : list) {
                        if (!h && !Double.isNaN(m.heartRate)) {
                            hr = String.valueOf(Math.round(m.heartRate));
                            h = true;
                        }
                        if (!b && !Double.isNaN(m.systolic)) {
                            bp = Math.round(m.systolic) + "/" + Math.round(m.diastolic);
                            b = true;
                        }
                        if (!r && !Double.isNaN(m.respiratoryRate)) {
                            rr = String.valueOf(Math.round(m.respiratoryRate));
                            r = true;
                        }
                        if (!s && !Double.isNaN(m.spo2)) {
                            sp = String.valueOf(Math.round(m.spo2));
                            s = true;
                        }
                    }
                } catch (RuntimeException ignored) {
                    // leave placeholders
                }
                final String fhr = hr, fbp = bp, frr = rr, fsp = sp;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (destroyed) {
                            return;
                        }
                        set(R.id.valueHeart, fhr, VitalTarget.HEART_RATE);
                        set(R.id.valueBp, fbp, VitalTarget.BLOOD_PRESSURE);
                        set(R.id.valueResp, frr, VitalTarget.RESPIRATORY_RATE);
                        set(R.id.valueSpo2, fsp, VitalTarget.SPO2);
                    }
                });
            }
        });
    }

    private void set(int id, String value, VitalTarget t) {
        TextView tv = findViewById(id);
        tv.setText(value);
        tv.setContentDescription("Latest " + t.displayName + ": " + value + (value.equals("--") ? "" : " " + t.unit));
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        io.shutdownNow();
        super.onDestroy();
    }

    @Override
    public void onBackPressed() {
        new AlertDialog.Builder(this)
                .setTitle("Exit HealthWatcher?")
                .setMessage("Do you want to close the app?")
                .setPositiveButton("Exit", new android.content.DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(android.content.DialogInterface d, int w) {
                        finishAffinity();
                    }
                })
                .setNegativeButton("Stay", null)
                .show();
    }
}
