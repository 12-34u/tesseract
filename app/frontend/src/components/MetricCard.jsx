import React from 'react';

export function MetricCard({ title, value, subtext, badge, status }) {
  return (
    <div className="card">
      <div className="card-title">
        <span>{title}</span>
        {badge && (
          <span className={`badge badge-${status || 'pass'}`}>
            {badge}
          </span>
        )}
      </div>
      <div className="card-value">{value}</div>
      {subtext && <div className="card-subtext">{subtext}</div>}
    </div>
  );
}
