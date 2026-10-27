package com.example.yo7a.healthwatcher;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.example.yo7a.healthwatcher.ml.VitalTarget;

import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Measurement history from the local database, with a trend chart per vital sign. */
public class HistoryActivity extends AppCompatActivity {
    private String user = "";
    private ExecutorService io;
    private volatile boolean destroyed = false;
    private List<MeasurementRecord> records = new ArrayList<>();
    private VitalTarget selected = VitalTarget.HEART_RATE;
    private TrendChartView chart;
    private ListView list;
    private TextView empty;
    private TextView summary;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_history);
        Bundle b = getIntent().getExtras();
        if (b != null && b.getString(MeasurementActivity.EXTRA_USER) != null) {
            user = b.getString(MeasurementActivity.EXTRA_USER);
        }
        io = Executors.newSingleThreadExecutor();
        chart = findViewById(R.id.trendChart);
        list = findViewById(R.id.historyList);
        empty = findViewById(R.id.historyEmpty);
        summary = findViewById(R.id.chartSummary);
        bindTab(R.id.tabHeart, VitalTarget.HEART_RATE);
        bindTab(R.id.tabBp, VitalTarget.BLOOD_PRESSURE);
        bindTab(R.id.tabResp, VitalTarget.RESPIRATORY_RATE);
        bindTab(R.id.tabSpo2, VitalTarget.SPO2);
        findViewById(R.id.clearHistory).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                confirmClear();
            }
        });
        load();
    }

    private void bindTab(int id, final VitalTarget t) {
        findViewById(id).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                selected = t;
                updateChart();
            }
        });
    }

    private void load() {
        io.execute(new Runnable() {
            @Override
            public void run() {
                List<MeasurementRecord> r;
                try {
                    r = new UserDB(HistoryActivity.this).getMeasurements(user, 500);
                } catch (RuntimeException e) {
                    r = new ArrayList<>();
                }
                final List<MeasurementRecord> fr = r;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (destroyed) {
                            return;
                        }
                        records = fr;
                        list.setAdapter(new RecordAdapter(records));
                        empty.setVisibility(records.isEmpty() ? View.VISIBLE : View.GONE);
                        updateChart();
                    }
                });
            }
        });
    }

    private void updateChart() {
        List<MeasurementRecord> asc = new ArrayList<>(records);
        Collections.reverse(asc);
        double[] a = new double[asc.size()];
        double[] d = selected == VitalTarget.BLOOD_PRESSURE ? new double[asc.size()] : null;
        int count = 0;
        for (int i = 0; i < asc.size(); i++) {
            MeasurementRecord m = asc.get(i);
            switch (selected) {
                case HEART_RATE:
                    a[i] = m.heartRate;
                    break;
                case RESPIRATORY_RATE:
                    a[i] = m.respiratoryRate;
                    break;
                case SPO2:
                    a[i] = m.spo2;
                    break;
                default:
                    a[i] = m.systolic;
                    d[i] = m.diastolic;
                    break;
            }
            if (!Double.isNaN(a[i])) {
                count++;
            }
        }
        chart.setData(a, d, ResultActivity.colorFor(selected), selected.unit);
        summary.setText(selected.displayName + " \u2013 " + count + " stored measurement" + (count == 1 ? "" : "s")
                + (selected == VitalTarget.BLOOD_PRESSURE ? " (systolic dark, diastolic grey)" : ""));
    }

    private void confirmClear() {
        new AlertDialog.Builder(this)
                .setTitle("Delete history?")
                .setMessage("All stored measurements of this account will be permanently deleted from this phone.")
                .setPositiveButton("Delete", new android.content.DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(android.content.DialogInterface dialog, int which) {
                        io.execute(new Runnable() {
                            @Override
                            public void run() {
                                try {
                                    new UserDB(HistoryActivity.this).deleteMeasurements(user);
                                } catch (RuntimeException ignored) {
                                    // reload shows the real state
                                }
                                load();
                            }
                        });
                    }
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private static String describe(MeasurementRecord m) {
        StringBuilder sb = new StringBuilder();
        if (!Double.isNaN(m.heartRate)) {
            sb.append("HR ").append(Math.round(m.heartRate)).append(" bpm [").append(m.sourceTag(VitalTarget.HEART_RATE)).append("]  ");
        }
        if (!Double.isNaN(m.systolic)) {
            sb.append("BP ").append(Math.round(m.systolic)).append('/').append(Math.round(m.diastolic))
                    .append(" [").append(m.sourceTag(VitalTarget.BLOOD_PRESSURE)).append("]  ");
        }
        if (!Double.isNaN(m.respiratoryRate)) {
            sb.append("RR ").append(Math.round(m.respiratoryRate)).append(" [").append(m.sourceTag(VitalTarget.RESPIRATORY_RATE)).append("]  ");
        }
        if (!Double.isNaN(m.spo2)) {
            sb.append("SpO\u2082 ").append(Math.round(m.spo2)).append("% [").append(m.sourceTag(VitalTarget.SPO2)).append("]");
        }
        return sb.toString().trim();
    }

    private final class RecordAdapter extends ArrayAdapter<MeasurementRecord> {
        RecordAdapter(List<MeasurementRecord> items) {
            super(HistoryActivity.this, 0, items);
        }

        @NonNull
        @Override
        public View getView(int position, View convertView, @NonNull ViewGroup parent) {
            View v = convertView != null ? convertView
                    : LayoutInflater.from(getContext()).inflate(R.layout.item_history, parent, false);
            MeasurementRecord m = getItem(position);
            ((TextView) v.findViewById(R.id.histDate)).setText(DateFormat.getDateTimeInstance(
                    DateFormat.MEDIUM, DateFormat.SHORT).format(new Date(m.timestampMillis)));
            ((TextView) v.findViewById(R.id.histValues)).setText(describe(m));
            TextView meta = v.findViewById(R.id.histMeta);
            String q = m.qualityStatus.equals("WARNINGS") ? "signal quality: warnings" : "signal quality: ok";
            meta.setText(String.format(Locale.US, "%s \u00b7 %.0f s \u00b7 %s", q, m.durationSec,
                    m.modelVersions.isEmpty() ? "no ML model" : "ML: " + m.modelVersions));
            meta.setTextColor(m.qualityStatus.equals("WARNINGS") ? Color.parseColor("#B54708") : Color.parseColor("#44505E"));
            return v;
        }
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
