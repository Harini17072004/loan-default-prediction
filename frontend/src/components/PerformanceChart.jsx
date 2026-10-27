import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CHART_METRICS } from "../constants/metrics";
import { formatPercentValue } from "../utils/format";

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>{label}</strong>
      <span>{formatPercentValue(Number(payload[0]?.value))}</span>
    </div>
  );
}

export default function PerformanceChart({ metrics }) {
  const data = CHART_METRICS.map(({ key, label, color }) => {
    const raw = metrics?.[key];
    return {
      key,
      label,
      color,
      value: Number.isFinite(raw) ? raw * 100 : null,
    };
  }).filter((item) => Number.isFinite(item.value));

  if (data.length === 0) {
    return <p className="chart-empty">Metrics are not available.</p>;
  }

  return (
    <div
      className="chart-canvas"
      role="img"
      aria-label="Bar chart of accuracy, precision, recall, F1, ROC-AUC, and PR-AUC from the API model information"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 18, bottom: 4, left: 8 }}
        >
          <CartesianGrid
            horizontal={false}
            stroke="#e2e8f0"
            strokeDasharray="4 4"
          />
          <XAxis
            type="number"
            domain={[0, 100]}
            ticks={[0, 25, 50, 75, 100]}
            tickFormatter={(value) => `${value}%`}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#64748b", fontSize: 12 }}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={86}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#334155", fontSize: 12, fontWeight: 600 }}
          />
          <Tooltip
            content={<ChartTooltip />}
            cursor={{ fill: "rgba(15, 118, 110, 0.06)" }}
          />
          <Bar
            dataKey="value"
            barSize={18}
            radius={[0, 8, 8, 0]}
            isAnimationActive={false}
          >
            {data.map((item) => (
              <Cell key={item.key} fill={item.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
