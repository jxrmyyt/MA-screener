"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const NICE_LABELS: Record<string, string> = {
  log_revenue: "Company size (log revenue)",
  sector_z_revenue_growth_yoy_pct: "Revenue growth (sector-relative)",
  sector_z_gross_margin_pct: "Gross margin (sector-relative)",
  sector_z_operating_margin_pct: "Operating margin (sector-relative)",
  sector_z_rd_expense_pct_of_revenue: "R&D intensity (sector-relative)",
  sector_z_debt_to_revenue: "Debt / revenue (sector-relative)",
  sector_z_cash_to_revenue: "Cash / revenue (sector-relative)",
  sector_z_net_debt_to_revenue: "Net debt / revenue (sector-relative)",
};

export function FeatureImportanceChart({ importance }: { importance: Record<string, number> }) {
  const data = Object.entries(importance)
    .map(([feature, value]) => ({ feature: NICE_LABELS[feature] ?? feature, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 12, fill: "#64748b" }} />
        <YAxis
          type="category"
          dataKey="feature"
          width={220}
          tick={{ fontSize: 12, fill: "#334155" }}
        />
        <Tooltip
          formatter={(value: number) => value.toFixed(3)}
          contentStyle={{ fontSize: 12, borderRadius: 8, borderColor: "#e2e8f0" }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((entry) => (
            <Cell key={entry.feature} fill={entry.value >= 0 ? "#2563eb" : "#f97316"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
