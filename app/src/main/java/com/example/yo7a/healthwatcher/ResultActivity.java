package com.example.yo7a.healthwatcher;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.yo7a.healthwatcher.ml.QualityReport;
import com.example.yo7a.healthwatcher.ml.ScreeningResult;
import com.example.yo7a.healthwatcher.ml.VitalEstimate;
import com.example.yo7a.healthwatcher.ml.VitalSignsReport;
import com.example.yo7a.healthwatcher.ml.VitalTarget;

import java.text.DateFormat;
import java.util.Date;

/**
 * Shared result screen. Each requested vital sign is shown with its value, the method that produced it
 * (signal processing vs validated ML), the comparison/experimental ML value if any, and an honest caveat.
 * The per-measurement subclasses only exist so the four separate result screens keep their own activities.
 */
public abstract class ResultActivity extends AppCompatActivity {
    private VitalSignsReport report;
    private String user = "";

    protected abstract String screenTitle();

    @SuppressWarnings("deprecation")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        String u = getIntent().getStringExtra(MeasurementActivity.EXTRA_USER);
        user = u == null ? "" : u;
        report = (VitalSignsReport) getIntent().getSerializableExtra(MeasurementActivity.EXTRA_REPORT);

        ((TextView) findViewById(R.id.resultTitle)).setText(screenTitle());
        findViewById(R.id.homeButton).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                goHome();
            }
        });
        findViewById(R.id.shareButton).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                share();
            }
        });
        if (report == null) {
            ((TextView) findViewById(R.id.resultSubtitle)).setText("No measurement data was received.");
            findViewById(R.id.shareButton).setEnabled(false);
            return;
        }
        ((TextView) findViewById(R.id.resultSubtitle)).setText("Measured "
                + DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(new Date(report.timestampMillis)));

        LinearLayout container = findViewById(R.id.resultContainer);
        LayoutInflater inflater = LayoutInflater.from(this);
        for (VitalEstimate e : report.all().values()) {
            container.addView(buildRow(inflater, container, e));
        }
        showScreening(report.screening);
        TextView q = findViewById(R.id.qualityNote);
        if (report.quality.hasWarnings()) {
            q.setVisibility(View.VISIBLE);
            q.setText("Signal-quality notice: " + join(report.quality.messages(QualityReport.Severity.WARNING)));
        } else {
            q.setVisibility(View.GONE);
        }
    }

    /** Shows the ML health-screening card (good / needs review / high priority) under the vital signs. */
    private void showScreening(ScreeningResult s) {
        View card = findViewById(R.id.screeningCard);
        if (s == null) {
            card.setVisibility(View.GONE);
            return;
        }
        card.setVisibility(View.VISIBLE);
        TextView verdict = (TextView) findViewById(R.id.screeningVerdict);
        TextView summary = (TextView) findViewById(R.id.screeningSummary);
        TextView findings = (TextView) findViewById(R.id.screeningFindings);
        TextView advice = (TextView) findViewById(R.id.screeningAdvice);
        TextView meta = (TextView) findViewById(R.id.screeningMeta);
        verdict.setText(s.headline());
        summary.setText(s.summary());
        card.setContentDescription("Health screening: " + s.headline() + ". " + s.summary());
        if (!s.available) {
            card.setBackgroundResource(R.drawable.bg_card);
            verdict.setTextColor(Color.parseColor("#44505E"));
            findings.setVisibility(View.GONE);
            advice.setVisibility(View.GONE);
            meta.setVisibility(View.GONE);
            return;
        }
        int bg, fg;
        switch (s.verdict) {
            case WITHIN_RANGES:
                bg = R.drawable.bg_ok;
                fg = Color.parseColor("#054F31");
                break;
            case NEEDS_REVIEW:
                bg = R.drawable.bg_warn;
                fg = Color.parseColor("#7A2E0E");
                break;
            default:
                bg = R.drawable.bg_error;
                fg = Color.parseColor("#912018");
                break;
        }
        card.setBackgroundResource(bg);
        card.setPadding(dp(16), dp(16), dp(16), dp(16));
        verdict.setTextColor(fg);
        StringBuilder sb = new StringBuilder();
        for (String f : s.findings) {
            sb.append(sb.length() > 0 ? "\n" : "").append("\u2022 ").append(f);
        }
        findings.setVisibility(sb.length() == 0 ? View.GONE : View.VISIBLE);
        findings.setText(sb.toString());
        advice.setVisibility(View.VISIBLE);
        advice.setText(s.advice());
        meta.setVisibility(View.VISIBLE);
        meta.setText(s.scoreLine() + (s.raisedByRangeCheck ? "\nRaised above the model's answer because a reading is outside the common range." : "")
                + "\n" + s.caveat());
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    private static String join(java.util.List<String> l) {
        StringBuilder sb = new StringBuilder();
        for (String s : l) {
            sb.append(sb.length() > 0 ? " " : "").append(s);
        }
        return sb.toString();
    }

    private View buildRow(LayoutInflater inflater, ViewGroup parent, VitalEstimate e) {
        View row = inflater.inflate(R.layout.item_result_row, parent, false);
        ((TextView) row.findViewById(R.id.rowLabel)).setText(e.target.displayName);
        TextView value = row.findViewById(R.id.rowValue);
        value.setText(e.displayValue());
        value.setTextColor(colorFor(e.target));
        ((TextView) row.findViewById(R.id.rowUnit)).setText(e.available ? e.target.unit : "");
        ((TextView) row.findViewById(R.id.rowBadge)).setText(e.sourceLabel());
        ((TextView) row.findViewById(R.id.rowCaution)).setText(e.caution());
        TextView ml = row.findViewById(R.id.rowMl);
        String note = e.mlNote();
        ml.setVisibility(note.isEmpty() ? View.GONE : View.VISIBLE);
        ml.setText(note);
        row.setContentDescription(e.target.displayName + ": " + e.displayValue() + " " + e.target.unit
                + ", " + e.sourceLabel());
        return row;
    }

    static int colorFor(VitalTarget t) {
        switch (t) {
            case HEART_RATE:
                return Color.parseColor("#B42318");
            case BLOOD_PRESSURE:
                return Color.parseColor("#2A4BB5");
            case RESPIRATORY_RATE:
                return Color.parseColor("#0B7A6C");
            default:
                return Color.parseColor("#6B2FA3");
        }
    }

    private void share() {
        StringBuilder sb = new StringBuilder("HealthWatcher measurement\n");
        for (VitalEstimate e : report.all().values()) {
            sb.append(e.target.displayName).append(": ").append(e.displayValue()).append(' ')
                    .append(e.available ? e.target.unit : "").append(" (").append(e.sourceLabel()).append(")\n");
        }
        if (report.screening != null) {
            sb.append('\n').append(report.screening.shareText()).append('\n');
        }
        sb.append("\nEstimates from a phone camera are for information only and are not a medical diagnosis.");
        Intent send = new Intent(Intent.ACTION_SEND);
        send.setType("text/plain");
        send.putExtra(Intent.EXTRA_SUBJECT, "HealthWatcher results");
        send.putExtra(Intent.EXTRA_TEXT, sb.toString());
        startActivity(Intent.createChooser(send, "Share results"));
    }

    private void goHome() {
        Intent i = new Intent(this, Primary.class);
        i.putExtra(MeasurementActivity.EXTRA_USER, user);
        i.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
        startActivity(i);
        finish();
    }

    @Override
    public void onBackPressed() {
        goHome();
    }
}
