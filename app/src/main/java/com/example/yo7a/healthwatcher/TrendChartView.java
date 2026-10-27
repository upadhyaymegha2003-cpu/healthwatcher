package com.example.yo7a.healthwatcher;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.View;

/** Minimal dependency-free line chart. Values are oldest-first; NaN entries are skipped. */
public class TrendChartView extends View {
    private final Paint axis = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint line = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint line2 = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint dot = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
    private double[] a = new double[0];
    private double[] b = null;
    private int color = Color.parseColor("#2A4BB5");
    private String emptyMessage = "No measurements yet";
    private String unit = "";

    public TrendChartView(Context c) {
        super(c);
        init();
    }

    public TrendChartView(Context c, AttributeSet s) {
        super(c, s);
        init();
    }

    private void init() {
        float d = getResources().getDisplayMetrics().density;
        axis.setColor(Color.parseColor("#C5CDD8"));
        axis.setStrokeWidth(1 * d);
        line.setStyle(Paint.Style.STROKE);
        line.setStrokeWidth(2.5f * d);
        line2.setStyle(Paint.Style.STROKE);
        line2.setStrokeWidth(2.5f * d);
        line2.setColor(Color.parseColor("#7A8798"));
        text.setColor(Color.parseColor("#44505E"));
        text.setTextSize(12 * getResources().getDisplayMetrics().scaledDensity);
    }

    /** @param second optional second series (e.g. diastolic), same length as first. */
    public void setData(double[] first, double[] second, int seriesColor, String unitLabel) {
        a = first == null ? new double[0] : first;
        b = second;
        color = seriesColor;
        unit = unitLabel == null ? "" : unitLabel;
        int n = 0;
        for (double v : a) {
            if (!Double.isNaN(v)) {
                n++;
            }
        }
        setContentDescription(n == 0 ? emptyMessage : "Trend chart with " + n + " measurements, unit " + unit);
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float d = getResources().getDisplayMetrics().density;
        float left = 44 * d, right = 12 * d, top = 12 * d, bottom = 24 * d;
        float w = getWidth() - left - right, h = getHeight() - top - bottom;
        canvas.drawLine(left, top, left, top + h, axis);
        canvas.drawLine(left, top + h, left + w, top + h, axis);

        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        int count = 0;
        for (int i = 0; i < a.length; i++) {
            double[] vs = {a[i], b != null && i < b.length ? b[i] : Double.NaN};
            for (double v : vs) {
                if (!Double.isNaN(v)) {
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                }
            }
            if (!Double.isNaN(a[i])) {
                count++;
            }
        }
        if (count == 0) {
            text.setTextAlign(Paint.Align.CENTER);
            canvas.drawText(emptyMessage, getWidth() / 2f, getHeight() / 2f, text);
            return;
        }
        if (max - min < 4) {
            double mid = (max + min) / 2;
            min = mid - 2;
            max = mid + 2;
        }
        double pad = (max - min) * 0.1;
        min -= pad;
        max += pad;
        text.setTextAlign(Paint.Align.RIGHT);
        canvas.drawText(String.valueOf(Math.round(max)), left - 4 * d, top + 10 * d, text);
        canvas.drawText(String.valueOf(Math.round(min)), left - 4 * d, top + h, text);
        text.setTextAlign(Paint.Align.LEFT);
        canvas.drawText(unit, 2 * d, top + h / 2, text);

        drawSeries(canvas, a, line, color, left, top, w, h, min, max, d);
        if (b != null) {
            drawSeries(canvas, b, line2, line2.getColor(), left, top, w, h, min, max, d);
        }
    }

    private void drawSeries(Canvas c, double[] s, Paint p, int col, float left, float top, float w, float h,
                            double min, double max, float d) {
        p.setColor(col);
        dot.setColor(col);
        Path path = new Path();
        boolean started = false;
        int n = s.length;
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(s[i])) {
                continue;
            }
            float x = left + (n == 1 ? w / 2 : w * i / (n - 1));
            float y = (float) (top + h - (s[i] - min) / (max - min) * h);
            if (!started) {
                path.moveTo(x, y);
                started = true;
            } else {
                path.lineTo(x, y);
            }
            c.drawCircle(x, y, 3.5f * d, dot);
        }
        c.drawPath(path, p);
    }
}
